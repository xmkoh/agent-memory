"""
LangGraph ``BaseStore`` backed by Weaviate.

This is a drop-in store for LangChain Deep Agents.  Construct it and hand it
straight to the real ``deepagents.create_deep_agent`` — no custom agent factory
required::

    import weaviate
    from deepagents import create_deep_agent
    from deepagents.backends import StoreBackend
    from weaviate_store import WeaviateStore

    client = weaviate.connect_to_local()
    store  = WeaviateStore(client)            # Weaviate is the persistence layer
    agent  = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=StoreBackend(),               # routes the filesystem tools to the store
        store=store,
    )

Required contract (langgraph.store.base.BaseStore)
--------------------------------------------------
The abstract surface is the two batch entrypoints, ``batch`` and ``abatch``;
LangGraph's concrete convenience methods (``get``/``put``/``search``/... and the
``a*`` variants) delegate to them.  The docs additionally specify the five async
operations as the contract a store must honour:

    aget(namespace, key)
    aput(namespace, key, value, index=None)
    adelete(namespace, key)
    asearch(namespace_prefix, *, query=None, filter=None, limit=10, offset=0)
    alist_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100, offset=0)

We implement all of the above plus their sync counterparts plus ``batch`` /
``abatch`` so the store works under both sync and async graph execution.  The
Weaviate Python client is synchronous, so the async methods offload the blocking
work to a thread (via ``asyncio.to_thread``) to avoid stalling the event loop.

Namespace tuples are serialised with the unit-separator byte (\\x1f), which will
not appear in normal namespace strings.

Vectors come from the same t2v-transformers inference service the rest of the
project uses, so no extra embedding dependency is required.  Pass a custom
``embedder`` callable to override (useful in tests or for a different model).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

import requests
import weaviate
from weaviate.classes.config import Configure, DataType, Property
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

_COLLECTION = "StoreItem"
_SEP = "\x1f"          # unit-separator — safe namespace delimiter
_T2V_DEFAULT = "http://t2v-transformers:8080"


def _ns_encode(namespace: tuple[str, ...]) -> str:
    return _SEP.join(namespace)


def _ns_decode(raw: str) -> tuple[str, ...]:
    return tuple(raw.split(_SEP)) if raw else ()


class WeaviateStore(BaseStore):
    """
    Weaviate-backed LangGraph store.

    Items are kept in a dedicated ``StoreItem`` collection with self-managed
    vectors (vectorizer=none).  ``put`` derives an embedding from the indexed
    fields so ``search`` can rank with a ``near_vector`` query; items written
    with ``index=False`` get no vector and are excluded from semantic search.
    """

    # BaseStore advertises whether the store supports TTL / vector search.
    supports_ttl = False

    def __init__(
        self,
        client: weaviate.WeaviateClient,
        embedder: Optional[Callable[[str], list[float]]] = None,
        t2v_url: str = _T2V_DEFAULT,
    ):
        self.client = client
        self._embedder = embedder
        self._t2v_url = t2v_url
        self._init_schema()
        self.col = self.client.collections.get(_COLLECTION)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        if self.client.collections.exists(_COLLECTION):
            return
        self.client.collections.create(
            name=_COLLECTION,
            properties=[
                Property(name="namespace", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="key",       data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="value",     data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="created_at", data_type=DataType.DATE, skip_vectorization=True),
                Property(name="updated_at", data_type=DataType.DATE, skip_vectorization=True),
            ],
            # We own all vectors; the transformers module is not used by the
            # collection, so it never needs re-creating if that module changes.
            vectorizer_config=Configure.Vectorizer.none(),
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*.

        Uses the plugged-in embedder if provided; otherwise calls the local
        t2v-transformers inference service that docker-compose already runs.
        """
        if self._embedder is not None:
            return self._embedder(text)
        resp = requests.post(
            f"{self._t2v_url}/vectors",
            json={"text": text},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["vector"]

    def _index_text(self, value: dict[str, Any], index: list[str] | bool | None) -> Optional[str]:
        """
        Return the text to embed for a value + index spec, or None if the item
        should not be vectorised.

        - index=False        → not indexed (no vector, excluded from search)
        - index=None / True  → embed all top-level string values
        - index=[...]        → embed the named string fields
        """
        if index is False:
            return None
        if index is None or index is True:
            parts = [str(v) for v in value.values() if isinstance(v, str) and v]
        else:
            parts = [str(value[k]) for k in index if k in value and isinstance(value[k], str)]
        return " ".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup(self, namespace: tuple[str, ...], key: str):
        """Return the raw Weaviate object for (namespace, key), or None."""
        response = self.col.query.fetch_objects(
            filters=(
                Filter.by_property("namespace").equal(_ns_encode(namespace))
                & Filter.by_property("key").equal(key)
            ),
            limit=1,
        )
        return response.objects[0] if response.objects else None

    @staticmethod
    def _to_item(obj) -> Item:
        p = obj.properties
        return Item(
            namespace=_ns_decode(p["namespace"]),
            key=p["key"],
            value=json.loads(p["value"]),
            created_at=p["created_at"],
            updated_at=p["updated_at"],
        )

    @staticmethod
    def _to_search_item(obj) -> SearchItem:
        p = obj.properties
        score = None
        if obj.metadata:
            score = obj.metadata.certainty if obj.metadata.certainty is not None else obj.metadata.score
        return SearchItem(
            namespace=_ns_decode(p["namespace"]),
            key=p["key"],
            value=json.loads(p["value"]),
            created_at=p["created_at"],
            updated_at=p["updated_at"],
            score=score,
        )

    def _ns_prefix_filter(self, namespace_prefix: tuple[str, ...]) -> Optional[Filter]:
        """A filter matching the exact namespace or any deeper sub-namespace."""
        if not namespace_prefix:
            return None
        encoded = _ns_encode(namespace_prefix)
        return (
            Filter.by_property("namespace").equal(encoded)
            | Filter.by_property("namespace").like(f"{encoded}{_SEP}*")
        )

    # ------------------------------------------------------------------
    # Sync implementation (the real Weaviate work lives here)
    # ------------------------------------------------------------------

    def get(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: Optional[bool] = None) -> Optional[Item]:
        obj = self._lookup(namespace, key)
        return self._to_item(obj) if obj is not None else None

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | bool | None = None,
        *,
        ttl: Optional[float] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        props = {
            "namespace": _ns_encode(namespace),
            "key": key,
            "value": json.dumps(value),
        }

        # Compute a vector only when the item should be searchable.
        vector = None
        text = self._index_text(value, index)
        if text:
            vector = self._embed(text)

        existing = self._lookup(namespace, key)
        if existing is not None:
            self.col.data.update(
                uuid=existing.uuid,
                properties={**props, "updated_at": now},
                vector=vector,
            )
        else:
            self.col.data.insert(
                uuid=uuid.uuid4(),
                properties={**props, "created_at": now, "updated_at": now},
                vector=vector,
            )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        obj = self._lookup(namespace, key)
        if obj is not None:
            self.col.data.delete_by_id(obj.uuid)

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
        ns_filter = self._ns_prefix_filter(namespace_prefix)

        if query is not None:
            response = self.col.query.near_vector(
                near_vector=self._embed(query),
                filters=ns_filter,
                limit=limit,
                offset=offset,
                return_metadata=MetadataQuery(certainty=True, score=True),
            )
        else:
            response = self.col.query.fetch_objects(
                filters=ns_filter,
                limit=limit,
                offset=offset,
            )

        items = [self._to_search_item(obj) for obj in response.objects]

        # Optional exact-match value filtering, applied client-side.
        if filter:
            items = [
                it for it in items
                if all(it.value.get(k) == v for k, v in filter.items())
            ]
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
        # Weaviate has no DISTINCT query, so we group via aggregate then decode.
        agg = self.col.aggregate.over_all(group_by="namespace")
        seen: set[tuple[str, ...]] = set()
        ordered: list[tuple[str, ...]] = []

        for group in agg.groups:
            ns = _ns_decode(group.grouped_by.value)
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

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | bool | None = None,
        *,
        ttl: Optional[float] = None,
    ) -> None:
        return await asyncio.to_thread(self.put, namespace, key, value, index)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        return await asyncio.to_thread(self.delete, namespace, key)

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
    ) -> list[SearchItem]:
        return await asyncio.to_thread(
            lambda: self.search(
                namespace_prefix, query=query, filter=filter, limit=limit, offset=offset
            )
        )

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        suffix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return await asyncio.to_thread(
            lambda: self.list_namespaces(
                prefix=prefix, suffix=suffix, max_depth=max_depth, limit=limit, offset=offset
            )
        )

    # ------------------------------------------------------------------
    # Batch entrypoints (the abstract surface of BaseStore)
    # ------------------------------------------------------------------

    def _dispatch(self, op: Op) -> Result:
        if isinstance(op, GetOp):
            return self.get(op.namespace, op.key)
        if isinstance(op, SearchOp):
            return self.search(
                op.namespace_prefix,
                query=op.query,
                filter=op.filter,
                limit=op.limit,
                offset=op.offset,
            )
        if isinstance(op, PutOp):
            # A PutOp with value=None is the delete convention.
            if op.value is None:
                self.delete(op.namespace, op.key)
            else:
                self.put(op.namespace, op.key, op.value, op.index)
            return None
        if isinstance(op, ListNamespacesOp):
            prefix = suffix = None
            for cond in (op.match_conditions or ()):
                if cond.match_type == "prefix":
                    prefix = cond.path
                elif cond.match_type == "suffix":
                    suffix = cond.path
            return self.list_namespaces(
                prefix=prefix,
                suffix=suffix,
                max_depth=op.max_depth,
                limit=op.limit,
                offset=op.offset,
            )
        raise TypeError(f"Unsupported store operation: {type(op).__name__}")

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return [self._dispatch(op) for op in ops]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))
