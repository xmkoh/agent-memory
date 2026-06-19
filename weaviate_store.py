"""
LangGraph ``BaseStore`` backed by the existing WeaviateMemoryManager.

This is a drop-in store for LangChain Deep Agents.  It wraps the existing
``WeaviateMemoryManager`` — the same ``Document`` and ``Chunk`` Weaviate
collections that power the file tools — rather than creating a separate store
collection, so all data stays in one place.

Mapping:
    namespace tuple → folder_path  (joined with "/")
    key             → filename
    value dict      → JSON-serialised raw_content
    search          → near_text on the Chunk collection (text2vec-transformers)

Usage::

    import weaviate
    from deepagents import create_deep_agent
    from deepagents.backends import StoreBackend
    from memory_manager import WeaviateMemoryManager
    from weaviate_store import WeaviateStore

    client  = weaviate.connect_to_local()
    manager = WeaviateMemoryManager(client)
    store   = WeaviateStore(manager)

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=StoreBackend(),
        store=store,
    )

Required contract (langgraph.store.base.BaseStore)
--------------------------------------------------
The five required async operations:

    aget / aput / adelete / asearch / alist_namespaces

plus their sync counterparts and the ``batch`` / ``abatch`` entrypoints.
The Weaviate client is synchronous, so async methods offload via
``asyncio.to_thread``.

Note: the ``Document`` schema has no ``updated_at`` field, so both
``Item.created_at`` and ``Item.updated_at`` are populated from ``created_at``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import weaviate.classes.query as wq
from weaviate.classes.query import Filter, MetadataQuery
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from memory_manager import WeaviateMemoryManager


def _ns_to_folder(namespace: tuple[str, ...]) -> str:
    return "/".join(namespace)


def _folder_to_ns(folder: str) -> tuple[str, ...]:
    return tuple(folder.split("/")) if folder else ()


class WeaviateStore(BaseStore):
    """
    LangGraph ``BaseStore`` that delegates to ``WeaviateMemoryManager``.

    The store uses the same ``Document`` (raw content) and ``Chunk``
    (vectorised) collections as the agent's file tools, so memories written
    via the store are immediately visible to ``read_file``, ``search_memory``,
    etc. and vice-versa.

    ``put`` writes directly to the underlying Weaviate collections, bypassing
    the conflict-check in ``write_markdown`` (which is designed for the
    read-then-edit tool workflow, not for unconditional store upserts).
    ``get`` likewise reads the collection directly so it does not record a
    hash snapshot that could interfere with the file-tool conflict detection.
    """

    supports_ttl = False

    def __init__(self, manager: WeaviateMemoryManager):
        self.manager = manager
        self._docs   = manager.document_collection
        self._chunks = manager.chunk_collection

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_obj(self, namespace: tuple[str, ...], key: str):
        """Return the raw Weaviate Document object for (namespace, key), or None."""
        response = self._docs.query.fetch_objects(
            filters=(
                Filter.by_property("folder_path").equal(_ns_to_folder(namespace))
                & Filter.by_property("filename").equal(key)
            ),
            limit=1,
        )
        return response.objects[0] if response.objects else None

    @staticmethod
    def _parse_value(raw_content: str) -> dict[str, Any]:
        """Deserialise stored content as JSON; fall back to a plain string dict."""
        try:
            v = json.loads(raw_content)
            if isinstance(v, dict):
                return v
        except (json.JSONDecodeError, TypeError):
            pass
        return {"content": raw_content}

    @staticmethod
    def _obj_to_item(obj, namespace: tuple[str, ...], key: str) -> Item:
        p = obj.properties
        ts = p.get("created_at") or datetime.now(timezone.utc)
        return Item(
            namespace=namespace,
            key=key,
            value=WeaviateStore._parse_value(p.get("raw_content", "")),
            created_at=ts,
            updated_at=ts,   # Document has no updated_at; reuse created_at
        )

    @staticmethod
    def _chunk_obj_to_search_item(chunk_obj) -> Optional[SearchItem]:
        """Convert a Chunk near-text result to a SearchItem using its parent Document."""
        refs = chunk_obj.references.get("hasDocument")
        if not refs or not refs.objects:
            return None
        parent = refs.objects[0]
        p = parent.properties
        folder = p.get("folder_path", "")
        ns = _folder_to_ns(folder)
        ts = p.get("created_at") or datetime.now(timezone.utc)
        score = None
        if chunk_obj.metadata:
            score = (chunk_obj.metadata.certainty
                     if chunk_obj.metadata.certainty is not None
                     else chunk_obj.metadata.score)
        return SearchItem(
            namespace=ns,
            key=p.get("filename", ""),
            value=WeaviateStore._parse_value(p.get("raw_content", "")),
            created_at=ts,
            updated_at=ts,
            score=score,
        )

    # ------------------------------------------------------------------
    # Sync implementation
    # ------------------------------------------------------------------

    def get(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: Optional[bool] = None) -> Optional[Item]:
        obj = self._lookup_obj(namespace, key)
        if obj is None:
            return None
        return self._obj_to_item(obj, namespace, key)

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | bool | None = None,
        *,
        ttl: Optional[float] = None,
    ) -> None:
        folder   = _ns_to_folder(namespace)
        content  = json.dumps(value)
        now      = datetime.now(timezone.utc).isoformat()
        existing = self._lookup_obj(namespace, key)

        if existing is not None:
            # Unconditional upsert — bypass the conflict check in write_markdown.
            self._docs.data.update(
                uuid=existing.uuid,
                properties={"raw_content": content},
            )
            doc_uuid = existing.uuid
            self.manager._delete_chunks_for(doc_uuid)
        else:
            doc_uuid = uuid.uuid4()
            self._docs.data.insert(
                uuid=doc_uuid,
                properties={
                    "filename":    key,
                    "folder_path": folder,
                    "raw_content": content,
                    "created_at":  now,
                },
            )

        # Rebuild the semantic chunks unless the caller explicitly opts out.
        if index is not False:
            chunks = self.manager._chunk_content(content)
            self.manager._insert_chunks(chunks, doc_uuid)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        obj = self._lookup_obj(namespace, key)
        if obj is not None:
            self.manager._delete_chunks_for(obj.uuid)
            self._docs.data.delete_by_id(obj.uuid)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
    ) -> list[SearchItem]:
        folder_prefix = _ns_to_folder(namespace_prefix)

        if query is not None:
            # Use near_text on the Chunk collection.  The text2vec-transformers
            # module embedded the chunks on write, so we get semantic ranking for free.
            ns_filter = (
                Filter.by_ref("hasDocument")
                      .by_property("folder_path")
                      .equal(folder_prefix)
                | Filter.by_ref("hasDocument")
                      .by_property("folder_path")
                      .like(f"{folder_prefix}/*")
            ) if folder_prefix else None

            response = self._chunks.query.near_text(
                query=query,
                filters=ns_filter,
                limit=limit + offset,
                return_metadata=MetadataQuery(certainty=True, score=True),
                return_references=[wq.QueryReference(link_on="hasDocument")],
            )

            # Deduplicate by parent document (a document may have many chunks).
            seen: dict[str, SearchItem] = {}
            for chunk_obj in response.objects:
                si = self._chunk_obj_to_search_item(chunk_obj)
                if si is None:
                    continue
                doc_key = f"{si.namespace}:{si.key}"
                if doc_key not in seen:
                    seen[doc_key] = si

            items: list[SearchItem] = list(seen.values())[offset: offset + limit]
        else:
            # No query: list documents whose folder_path starts with the prefix.
            ns_filter = (
                Filter.by_property("folder_path").equal(folder_prefix)
                | Filter.by_property("folder_path").like(f"{folder_prefix}/*")
            ) if folder_prefix else None

            response = self._docs.query.fetch_objects(
                filters=ns_filter,
                limit=limit,
                offset=offset,
            )
            items = []
            for obj in response.objects:
                p  = obj.properties
                ns = _folder_to_ns(p.get("folder_path", ""))
                ts = p.get("created_at") or datetime.now(timezone.utc)
                items.append(SearchItem(
                    namespace=ns,
                    key=p.get("filename", ""),
                    value=self._parse_value(p.get("raw_content", "")),
                    created_at=ts,
                    updated_at=ts,
                    score=None,
                ))

        if filter:
            items = [it for it in items
                     if all(it.value.get(k) == v for k, v in filter.items())]
        return items

    def list_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        suffix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        seen: set[tuple[str, ...]] = set()
        ordered: list[tuple[str, ...]] = []

        for folder in self.manager.list_folders():
            ns = _folder_to_ns(folder)
            if prefix and ns[: len(prefix)] != prefix:
                continue
            if suffix and ns[-len(suffix):] != suffix:
                continue
            trimmed = ns[:max_depth] if max_depth is not None else ns
            if trimmed not in seen:
                seen.add(trimmed)
                ordered.append(trimmed)

        return ordered[offset: offset + limit]

    # ------------------------------------------------------------------
    # Async implementation (required contract) — offload blocking I/O
    # ------------------------------------------------------------------

    async def aget(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: Optional[bool] = None) -> Optional[Item]:
        return await asyncio.to_thread(self.get, namespace, key)

    async def aput(self, namespace: tuple[str, ...], key: str, value: dict[str, Any],
                   index: list[str] | bool | None = None, *, ttl: Optional[float] = None) -> None:
        await asyncio.to_thread(self.put, namespace, key, value, index)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        await asyncio.to_thread(self.delete, namespace, key)

    async def asearch(self, namespace_prefix: tuple[str, ...], *, query: Optional[str] = None,
                      filter: Optional[dict[str, Any]] = None, limit: int = 10,
                      offset: int = 0, refresh_ttl: Optional[bool] = None) -> list[SearchItem]:
        return await asyncio.to_thread(
            lambda: self.search(namespace_prefix, query=query, filter=filter, limit=limit, offset=offset)
        )

    async def alist_namespaces(self, *, prefix: Optional[tuple[str, ...]] = None,
                                suffix: Optional[tuple[str, ...]] = None,
                                max_depth: Optional[int] = None, limit: int = 100,
                                offset: int = 0) -> list[tuple[str, ...]]:
        return await asyncio.to_thread(
            lambda: self.list_namespaces(prefix=prefix, suffix=suffix,
                                         max_depth=max_depth, limit=limit, offset=offset)
        )

    # ------------------------------------------------------------------
    # Batch entrypoints (the abstract surface of BaseStore)
    # ------------------------------------------------------------------

    def _dispatch(self, op: Op) -> Result:
        if isinstance(op, GetOp):
            return self.get(op.namespace, op.key)
        if isinstance(op, PutOp):
            if op.value is None:
                self.delete(op.namespace, op.key)
            else:
                self.put(op.namespace, op.key, op.value, op.index)
            return None
        if isinstance(op, SearchOp):
            return self.search(
                op.namespace_prefix, query=op.query, filter=op.filter,
                limit=op.limit, offset=op.offset,
            )
        if isinstance(op, ListNamespacesOp):
            prefix = suffix = None
            for cond in (op.match_conditions or ()):
                if cond.match_type == "prefix":
                    prefix = cond.path
                elif cond.match_type == "suffix":
                    suffix = cond.path
            return self.list_namespaces(
                prefix=prefix, suffix=suffix,
                max_depth=op.max_depth, limit=op.limit, offset=op.offset,
            )
        raise TypeError(f"Unsupported store operation: {type(op).__name__}")

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return [self._dispatch(op) for op in ops]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))
