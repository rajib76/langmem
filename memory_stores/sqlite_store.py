import sqlite3
import json
import math
import asyncio
from datetime import datetime
from typing import Any, Iterable, Optional, Union, List

from langgraph.store.base import BaseStore, Item, GetOp, PutOp, SearchOp, ListNamespacesOp, SearchItem
# Imports from the basestore module.
# Assumes BaseStore, Item, SearchItem, GetOp, PutOp, SearchOp, ListNamespacesOp,
# MatchCondition, NOT_PROVIDED have already been imported.
from langgraph.store.base.embed import ensure_embeddings, get_text_at_path


# Optionally, import your IndexConfig type if needed:
# from your_module import IndexConfig

class SQLLITESTORE(BaseStore):
    """
    A SQLite-backed key-value store with semantic indexing support.

    Items are stored in a table with the following columns:
      - ns: A text field representing the joined namespace (joined by '/')
      - namespace_json: A JSON-encoded version of the namespace tuple
      - key: The unique key for the item
      - value_json: JSON-encoded dictionary of the stored value
      - embedding: JSON-encoded list of floats (the embedding vector)
      - created_at: ISO formatted creation timestamp
      - updated_at: ISO formatted update timestamp
    """
    # This implementation does not support TTL.
    supports_ttl: bool = False

    def __init__(self, db_path: str = "memstore.sqlite", index_config: Optional[dict] = None):
        """
        Args:
            db_path: Path to the SQLite database file (":memory:" for in-memory DB).
            index_config: Optional configuration for semantic indexing.
                Should include at least:
                  - "dims": number of dimensions
                  - "embed": an embedding function or provider,
                  - "fields": list of JSON-path fields to extract text from (default ["$"])
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.index_config = index_config
        self._create_table()

    def _create_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS store (
                ns TEXT NOT NULL,
                namespace_json TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                embedding TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (ns, key)
            )
        """)
        self.conn.commit()

    def _ns_from_tuple(self, namespace: tuple[str, ...]) -> str:
        """Join namespace tuple into a single string."""
        return "/".join(namespace)

    def _make_item_from_row(self, row: sqlite3.Row) -> Item:
        return Item(
            value=json.loads(row["value_json"]),
            key=row["key"],
            namespace=tuple(json.loads(row["namespace_json"])),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def compute_embedding(self, value: dict[str, Any], index_override: Optional[list[str]]) -> list[float] | None:
        """
        Compute the embedding for the given value.

        Args:
            value: The dictionary to compute embedding for.
            index_override: If provided, list of fields to extract text from.
                Otherwise, falls back to self.index_config["fields"] or ["$"].

        Returns:
            JSON encoded embedding vector, or None if no text found.
        """
        if self.index_config is None:
            return None
        fields = index_override if index_override is not None else self.index_config.get("fields", ["$"])
        texts = []
        for field in fields:
            extracted = get_text_at_path(value, field)
            if extracted:
                texts.append(str(extracted))
        if not texts:
            return None
        full_text = "\n".join(texts)
        embed_fn = ensure_embeddings(self.index_config["embed"])
        embedding_vector = embed_fn.embed_documents([full_text])[0]
        # Compute embedding vector; assume embed_fn accepts a list of texts.
        # embedding_vector = embed_fn([full_text])[0]
        return json.dumps(embedding_vector)

    def batch(self, ops: Iterable[Union[GetOp, PutOp, SearchOp, ListNamespacesOp]]) -> List[Optional[Any]]:
        results: List[Optional[Any]] = []
        for op in ops:
            print("op ", op)
            if isinstance(op, GetOp):
                ns = self._ns_from_tuple(op.namespace)
                cur = self.conn.execute("SELECT * FROM store WHERE ns = ? AND key = ?", (ns, op.key))
                row = cur.fetchone()
                results.append(self._make_item_from_row(row) if row else None)

            elif isinstance(op, PutOp):
                ns = self._ns_from_tuple(op.namespace)
                namespace_json = json.dumps(list(op.namespace))
                now = datetime.now().isoformat()
                if op.value is None:
                    # Delete the item
                    self.conn.execute("DELETE FROM store WHERE ns = ? AND key = ?", (ns, op.key))
                    embedding_json = None
                else:
                    # Compute embedding if indexing is enabled
                    if self.index_config is not None and op.index is not False:
                        embedding = self.compute_embedding(op.value, op.index)
                        print("embedd ", embedding)
                    else:
                        embedding = None
                    # Preserve created_at if updating
                    cur = self.conn.execute("SELECT created_at FROM store WHERE ns = ? AND key = ?", (ns, op.key))
                    row = cur.fetchone()
                    created_at = row["created_at"] if row else now
                    value_json = json.dumps(op.value)
                    self.conn.execute("""
                        INSERT INTO store (ns, namespace_json, key, value_json, embedding, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ns, key) DO UPDATE SET
                            value_json = excluded.value_json,
                            embedding = excluded.embedding,
                            updated_at = excluded.updated_at
                    """, (ns, namespace_json, op.key, value_json, embedding, created_at, now))
                self.conn.commit()
                results.append(None)

            elif isinstance(op, SearchOp):
                # Retrieve rows matching the namespace prefix.
                prefix_ns = self._ns_from_tuple(op.namespace_prefix)
                query_sql = "SELECT * FROM store WHERE ns = ? OR ns LIKE ?"
                like_pattern = prefix_ns + "/%"
                cur = self.conn.execute(query_sql, (prefix_ns, like_pattern))
                rows = cur.fetchall()
                items: List[Any] = []
                if op.query and self.index_config is not None:
                    # Semantic search: compute embedding for the query.
                    embed_fn = ensure_embeddings(self.index_config["embed"])
                    # query_embedding = embed_fn([op.query])[0]
                    query_embedding = embed_fn.embed_documents([op.query])[0]
                    for row in rows:
                        value = json.loads(row["value_json"])
                        # Apply filter if provided (simple equality on top-level keys)
                        if op.filter:
                            if any(value.get(k) != v for k, v in op.filter.items()):
                                continue
                        embedding_data = row["embedding"]
                        if embedding_data:
                            candidate_embedding = json.loads(embedding_data)
                            score = self.cosine_similarity(query_embedding, candidate_embedding)
                        else:
                            score = 0.0
                        item = SearchItem(
                            namespace=tuple(json.loads(row["namespace_json"])),
                            key=row["key"],
                            value=value,
                            created_at=datetime.fromisoformat(row["created_at"]),
                            updated_at=datetime.fromisoformat(row["updated_at"]),
                            score=score
                        )
                        items.append(item)
                    # Sort items by similarity score (higher is better)
                    items.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
                else:
                    # Fallback: perform substring search on the stored JSON value.
                    for row in rows:
                        value = json.loads(row["value_json"])
                        if op.filter:
                            if any(value.get(k) != v for k, v in op.filter.items()):
                                continue
                        if op.query:
                            if op.query.lower() not in json.dumps(value).lower():
                                continue
                        item = SearchItem(
                            namespace=tuple(json.loads(row["namespace_json"])),
                            key=row["key"],
                            value=value,
                            created_at=datetime.fromisoformat(row["created_at"]),
                            updated_at=datetime.fromisoformat(row["updated_at"]),
                            score=None
                        )
                        items.append(item)
                # Apply offset and limit.
                results.append(items[op.offset: op.offset + op.limit])

            elif isinstance(op, ListNamespacesOp):
                cur = self.conn.execute("SELECT DISTINCT namespace_json FROM store")
                rows = cur.fetchall()
                namespaces: List[tuple[str, ...]] = []
                for row in rows:
                    ns_tuple = tuple(json.loads(row["namespace_json"]))
                    if op.match_conditions:
                        matched = True
                        for cond in op.match_conditions:
                            if cond.match_type == "prefix":
                                if ns_tuple[:len(cond.path)] != cond.path:
                                    matched = False
                                    break
                            elif cond.match_type == "suffix":
                                if ns_tuple[-len(cond.path):] != cond.path:
                                    matched = False
                                    break
                        if not matched:
                            continue
                    if op.max_depth is not None:
                        ns_tuple = ns_tuple[:op.max_depth]
                    if ns_tuple not in namespaces:
                        namespaces.append(ns_tuple)
                results.append(namespaces[op.offset: op.offset + op.limit])
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Union[GetOp, PutOp, SearchOp, ListNamespacesOp]]) -> List[Optional[Any]]:
        return await asyncio.to_thread(self.batch, ops)
