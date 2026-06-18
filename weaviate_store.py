"""
LangGraph BaseStore backed by Weaviate.

Namespace tuples are serialised with US (\\x1f) as a separator — a non-printable
byte that won't appear in normal namespace strings.

Vectors come from the same t2v-transformers inference service that the rest of
the project uses, so no extra embedding dependency is required.  Pass a custom
``embedder`` callable to override (useful in tests or when a different model is
preferred).

Usage::

    from weaviate_store import WeaviateStore
    import weaviate

    client = weaviate.connect_to_local()
    store  = WeaviateStore(client)
    agent  = create_deep_agent(store=store)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from langgraph.store.base import BaseStore, Item, SearchItem

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

    put() stores items and computes their vector from indexed fields so that
    search() can rank results with a near_vector query.  Items stored without
    an index (index=False) get no vector and are excluded from semantic search.
    """

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
            # We own all vectors; the transformers module is not used here so
            # the collection does not need to be re-created if the module changes.
            vectorizer_config=Configure.Vectorizer.none(),
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*.

        Uses the plugged-in embedder if provided; falls back to the local
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
        Return the text to embed for a given value + index spec, or None when
        the item should not be vectorised.

        - index=False / None → not indexed
        - index=True         → join all top-level string values
        - index=[…]          → join the named string fields
        """
        if not index:
            return None
        if index is True:
            parts = [str(v) for v in value.values() if isinstance(v, str) and v]
        else:
            parts = [str(value[k]) for k in index if k in value and isinstance(value[k], str)]
        return " ".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ns_filter(self, namespace: tuple[str, ...]) -> Filter:
        return (
            Filter.by_property("namespace").equal(_ns_encode(namespace))
            & Filter.by_property("key").equal  # filled by callers
        )

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
            score = obj.metadata.certainty or obj.metadata.score
        return SearchItem(
            namespace=_ns_decode(p["namespace"]),
            key=p["key"],
            value=json.loads(p["value"]),
            created_at=p["created_at"],
            updated_at=p["updated_at"],
            score=score,
        )

    # ------------------------------------------------------------------
    # BaseStore interface
    # ------------------------------------------------------------------

    def get(self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        obj = self._lookup(namespace, key)
        return self._to_item(obj) if obj is not None else None

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | bool | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        props = {
            "namespace": _ns_encode(namespace),
            "key": key,
            "value": json.dumps(value),
        }

        # Compute vector only when the item should be searchable.
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

    def list_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        suffix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
    ) -> list[tuple[str, ...]]:
        # Weaviate has no DISTINCT query, so we group via aggregate then decode.
        agg = self.col.aggregate.over_all(group_by="namespace")
        seen: set[tuple[str, ...]] = set()
        results: list[tuple[str, ...]] = []

        for group in agg.groups:
            ns = _ns_decode(group.grouped_by.value)

            if prefix and ns[: len(prefix)] != prefix:
                continue
            if suffix and ns[-len(suffix) :] != suffix:
                continue

            trimmed = ns[:max_depth] if max_depth is not None else ns
            if trimmed not in seen:
                seen.add(trimmed)
                results.append(trimmed)
            if len(results) >= limit:
                break

        return results

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        # Build a Weaviate filter that restricts to the requested namespace prefix.
        if namespace_prefix:
            encoded_prefix = _ns_encode(namespace_prefix)
            # Exact match covers single-part namespaces; LIKE covers deeper ones.
            ns_filter = Filter.by_property("namespace").equal(encoded_prefix) | \
                        Filter.by_property("namespace").like(f"{encoded_prefix}{_SEP}*")
        else:
            ns_filter = None

        if query is not None:
            query_vector = self._embed(query)
            response = self.col.query.near_vector(
                near_vector=query_vector,
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

        return [self._to_search_item(obj) for obj in response.objects]
