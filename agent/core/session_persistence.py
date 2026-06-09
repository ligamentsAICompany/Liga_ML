"""Optional durable session persistence for the hosted backend.

The public CLI must keep working without MongoDB.  This module therefore
exposes one small async store interface and returns a no-op implementation
unless ``MONGODB_URI`` is configured and reachable.
"""

from __future__ import annotations

import logging
import os
import math
import re
from datetime import UTC, datetime
from typing import Any

from bson import BSON
from pymongo import AsyncMongoClient, DeleteMany, ReturnDocument, UpdateOne
from pymongo.errors import DuplicateKeyError, InvalidDocument, PyMongoError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_BSON_BYTES = 15 * 1024 * 1024
NO_DURABLE_STORE_WARNING = (
    "MONGODB_URI is not configured; sessions will not survive restarts"
)


def _now() -> datetime:
    return datetime.now(UTC)


def _doc_id(session_id: str, idx: int) -> str:
    return f"{session_id}:{idx}"


def _hf_job_slug(value: Any) -> str:
    return str(value or "").strip().rstrip("/").split("/")[-1]


def _same_hf_job_id(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    return left_text == right_text or _hf_job_slug(left_text) == _hf_job_slug(
        right_text
    )


def _hf_job_slug_regex(value: Any) -> dict[str, str] | None:
    slug = _hf_job_slug(value)
    if not slug:
        return None
    return {"$regex": rf"(^|/){re.escape(slug)}/?$"}


def _find_existing_response_row(
    candidates: list[dict[str, Any]], row: dict[str, Any], row_user_id: str
) -> dict[str, Any]:
    platform = str(row.get("platform") or "")
    job_id = str(row.get("job_id") or "")
    session_id = str(row.get("session_id") or "")

    def platform_matches(doc: dict[str, Any]) -> bool:
        return str(doc.get("platform") or "") == platform

    def exact_job_matches(doc: dict[str, Any]) -> bool:
        return str(doc.get("job_id") or "") == job_id

    def hf_job_matches(doc: dict[str, Any]) -> bool:
        return platform == "hf-jobs" and _same_hf_job_id(doc.get("job_id"), job_id)

    matchers = (
        lambda doc: (
            str(doc.get("user_id") or "") == row_user_id
            and platform_matches(doc)
            and exact_job_matches(doc)
        ),
        lambda doc: (
            str(doc.get("user_id") or "") == row_user_id
            and platform_matches(doc)
            and hf_job_matches(doc)
        ),
        lambda doc: (
            session_id
            and str(doc.get("session_id") or "") == session_id
            and platform_matches(doc)
            and exact_job_matches(doc)
        ),
        lambda doc: (
            session_id
            and str(doc.get("session_id") or "") == session_id
            and platform_matches(doc)
            and hf_job_matches(doc)
        ),
    )
    for matcher in matchers:
        for doc in candidates:
            if matcher(doc):
                return doc
    return {}


def _safe_message_doc(message: dict[str, Any]) -> dict[str, Any]:
    """Return a Mongo-safe message document payload.

    Mongo's hard document limit is 16 MB.  We stay below that and store an
    explicit marker rather than failing the whole snapshot for one huge tool log.
    """
    try:
        if len(BSON.encode({"message": message})) <= MAX_BSON_BYTES:
            return message
    except (InvalidDocument, OverflowError):
        pass
    return {
        "role": "tool",
        "content": (
            "[SYSTEM: A single persisted message exceeded MongoDB's document "
            "size/encoding limit and was replaced by this marker.]"
        ),
        "ml_intern_persistence_error": "message_too_large_or_invalid",
    }


class NoopSessionStore:
    """Async no-op store used when Mongo is not configured."""

    enabled = False

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_session(self, **_: Any) -> None:
        return None

    async def save_snapshot(self, **_: Any) -> None:
        return None

    async def load_session(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def list_sessions(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def soft_delete_session(self, *_: Any, **__: Any) -> None:
        return None

    async def update_session_fields(self, *_: Any, **__: Any) -> None:
        return None

    async def append_event(self, *_: Any, **__: Any) -> int | None:
        return None

    async def load_events_after(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def append_trace_message(self, *_: Any, **__: Any) -> int | None:
        return None

    async def load_trace_messages(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def get_quota(self, *_: Any, **__: Any) -> int | None:
        return None

    async def try_increment_quota(self, *_: Any, **__: Any) -> int | None:
        return None

    async def refund_quota(self, *_: Any, **__: Any) -> None:
        return None

    async def mark_pro_seen(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def upsert_response_rows(self, *_: Any, **__: Any) -> None:
        return None

    async def list_response_rows(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "rows": [],
            "page": 1,
            "page_size": 50,
            "total_rows": 0,
            "total_pages": 0,
            "has_next": False,
            "has_previous": False,
        }

    async def get_response_summary(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "total_responses": 0,
            "visible_count": 0,
            "batch_number": 1,
            "has_rows": False,
            "button_enabled": True,
            "durable": False,
            "store_type": "memory",
        }


class MongoSessionStore(NoopSessionStore):
    """MongoDB-backed session store."""

    enabled = True

    def __init__(self, uri: str, db_name: str) -> None:
        self.uri = uri
        self.db_name = db_name
        self.enabled = False
        self.client: AsyncMongoClient | None = None
        self.db = None

    async def init(self) -> None:
        try:
            self.client = AsyncMongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self.db = self.client[self.db_name]
            await self.client.admin.command("ping")
            await self._create_indexes()
            self.enabled = True
            logger.info("Mongo session persistence enabled (db=%s)", self.db_name)
        except Exception as e:
            logger.warning("Mongo session persistence disabled: %s", e)
            self.enabled = False
            if self.client is not None:
                await self.client.close()
            self.client = None
            self.db = None

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
        self.client = None
        self.db = None

    async def _create_indexes(self) -> None:
        if self.db is None:
            return
        await self.db.sessions.create_index(
            [("user_id", 1), ("visibility", 1), ("updated_at", -1)]
        )
        await self.db.sessions.create_index(
            [("visibility", 1), ("status", 1), ("last_active_at", -1)]
        )
        await self.db.session_messages.create_index(
            [("session_id", 1), ("idx", 1)], unique=True
        )
        await self.db.session_events.create_index(
            [("session_id", 1), ("seq", 1)], unique=True
        )
        await self.db.session_trace_messages.create_index(
            [("session_id", 1), ("seq", 1)], unique=True
        )
        await self.db.session_trace_messages.create_index([("created_at", -1)])
        await self.db.pro_users.create_index([("first_seen_pro_at", -1)])
        await self.db.response_rows.create_index(
            [("user_id", 1), ("actual_sequence_number", -1)]
        )
        await self.db.response_rows.create_index([("platform", 1), ("progress", 1)])
        await self.db.response_rows.create_index([("session_id", 1)])
        await self.db.response_rows.create_index([("job_id", 1)])

    def _ready(self) -> bool:
        return bool(self.enabled and self.db is not None)

    async def upsert_session(
        self,
        *,
        session_id: str,
        user_id: str,
        model: str,
        title: str | None = None,
        surface: str = "frontend",
        created_at: datetime | None = None,
        runtime_state: str = "idle",
        status: str = "active",
        message_count: int = 0,
        turn_count: int = 0,
        pending_approval: list[dict[str, Any]] | None = None,
        claude_counted: bool = False,
        notification_destinations: list[str] | None = None,
        auto_approval_enabled: bool = False,
        auto_approval_cost_cap_usd: float | None = None,
        auto_approval_estimated_spend_usd: float = 0.0,
        cloud_provider: str = "hf-jobs",
        training_goal: str = "agent-decide",
        output_policy: str = "cloud-and-hf-hub",
        uploaded_datasets: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._ready():
            return
        now = _now()
        await self.db.sessions.update_one(
            {"_id": session_id},
            {
                "$setOnInsert": {
                    "_id": session_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "surface": surface,
                    "created_at": created_at or now,
                    "schema_version": SCHEMA_VERSION,
                    "visibility": "live",
                },
                "$set": {
                    "title": title,
                    "model": model,
                    "status": status,
                    "runtime_state": runtime_state,
                    "updated_at": now,
                    "last_active_at": now,
                    "message_count": message_count,
                    "turn_count": turn_count,
                    "pending_approval": pending_approval or [],
                    "claude_counted": claude_counted,
                    "notification_destinations": notification_destinations or [],
                    "auto_approval_enabled": auto_approval_enabled,
                    "auto_approval_cost_cap_usd": auto_approval_cost_cap_usd,
                    "auto_approval_estimated_spend_usd": auto_approval_estimated_spend_usd,
                    "cloud_provider": cloud_provider,
                    "training_goal": training_goal,
                    "output_policy": output_policy,
                    "uploaded_datasets": uploaded_datasets or [],
                },
            },
            upsert=True,
        )

    async def save_snapshot(
        self,
        *,
        session_id: str,
        user_id: str,
        model: str,
        messages: list[dict[str, Any]],
        title: str | None = None,
        runtime_state: str = "idle",
        status: str = "active",
        turn_count: int = 0,
        pending_approval: list[dict[str, Any]] | None = None,
        claude_counted: bool = False,
        created_at: datetime | None = None,
        notification_destinations: list[str] | None = None,
        auto_approval_enabled: bool = False,
        auto_approval_cost_cap_usd: float | None = None,
        auto_approval_estimated_spend_usd: float = 0.0,
        cloud_provider: str = "hf-jobs",
        training_goal: str = "agent-decide",
        output_policy: str = "cloud-and-hf-hub",
        uploaded_datasets: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._ready():
            return
        now = _now()
        await self.upsert_session(
            session_id=session_id,
            user_id=user_id,
            model=model,
            title=title,
            created_at=created_at,
            runtime_state=runtime_state,
            status=status,
            message_count=len(messages),
            turn_count=turn_count,
            pending_approval=pending_approval,
            claude_counted=claude_counted,
            notification_destinations=notification_destinations,
            auto_approval_enabled=auto_approval_enabled,
            auto_approval_cost_cap_usd=auto_approval_cost_cap_usd,
            auto_approval_estimated_spend_usd=auto_approval_estimated_spend_usd,
            cloud_provider=cloud_provider,
            training_goal=training_goal,
            output_policy=output_policy,
            uploaded_datasets=uploaded_datasets,
        )
        ops: list[Any] = []
        for idx, raw in enumerate(messages):
            ops.append(
                UpdateOne(
                    {"_id": _doc_id(session_id, idx)},
                    {
                        "$set": {
                            "session_id": session_id,
                            "idx": idx,
                            "message": _safe_message_doc(raw),
                            "updated_at": now,
                        },
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            )
        ops.append(
            DeleteMany({"session_id": session_id, "idx": {"$gte": len(messages)}})
        )
        try:
            if ops:
                await self.db.session_messages.bulk_write(ops, ordered=False)
        except PyMongoError as e:
            logger.warning("Failed to persist session %s snapshot: %s", session_id, e)

    async def load_session(
        self, session_id: str, *, include_deleted: bool = False
    ) -> dict[str, Any] | None:
        if not self._ready():
            return None
        meta = await self.db.sessions.find_one({"_id": session_id})
        if not meta:
            return None
        if meta.get("visibility") == "deleted" and not include_deleted:
            return None
        cursor = self.db.session_messages.find({"session_id": session_id}).sort(
            "idx", 1
        )
        messages = [row.get("message") async for row in cursor]
        return {"metadata": meta, "messages": messages}

    async def list_sessions(
        self, user_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return []
        query: dict[str, Any] = {"user_id": user_id}
        if user_id == "dev":
            query = {}
        if not include_deleted:
            query["visibility"] = {"$ne": "deleted"}
        cursor = self.db.sessions.find(query).sort("updated_at", -1)
        return [row async for row in cursor]

    async def soft_delete_session(self, session_id: str) -> None:
        if not self._ready():
            return
        await self.db.sessions.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "visibility": "deleted",
                    "runtime_state": "idle",
                    "updated_at": _now(),
                }
            },
        )

    async def update_session_fields(self, session_id: str, **fields: Any) -> None:
        if not self._ready() or not fields:
            return
        fields["updated_at"] = _now()
        await self.db.sessions.update_one({"_id": session_id}, {"$set": fields})

    async def _next_seq(self, counter_id: str) -> int:
        doc = await self.db.counters.find_one_and_update(
            {"_id": counter_id},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["seq"])

    async def append_event(
        self, session_id: str, event_type: str, data: dict[str, Any] | None
    ) -> int | None:
        if not self._ready():
            return None
        try:
            seq = await self._next_seq(f"event:{session_id}")
            await self.db.session_events.insert_one(
                {
                    "_id": _doc_id(session_id, seq),
                    "session_id": session_id,
                    "seq": seq,
                    "event_type": event_type,
                    "data": data or {},
                    "created_at": _now(),
                }
            )
            return seq
        except PyMongoError as e:
            logger.debug("Failed to append event for %s: %s", session_id, e)
            return None

    async def load_events_after(
        self, session_id: str, after_seq: int = 0
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return []
        cursor = self.db.session_events.find(
            {"session_id": session_id, "seq": {"$gt": int(after_seq or 0)}}
        ).sort("seq", 1)
        return [row async for row in cursor]

    async def append_trace_message(
        self, session_id: str, message: dict[str, Any], source: str = "message"
    ) -> int | None:
        if not self._ready():
            return None
        try:
            seq = await self._next_seq(f"trace:{session_id}")
            await self.db.session_trace_messages.insert_one(
                {
                    "_id": _doc_id(session_id, seq),
                    "session_id": session_id,
                    "seq": seq,
                    "role": message.get("role"),
                    "message": _safe_message_doc(message),
                    "source": source,
                    "created_at": _now(),
                }
            )
            return seq
        except PyMongoError as e:
            logger.debug("Failed to append trace message for %s: %s", session_id, e)
            return None

    async def load_trace_messages(self, session_id: str) -> list[dict[str, Any]]:
        if not self._ready():
            return []
        cursor = self.db.session_trace_messages.find({"session_id": session_id}).sort(
            "seq", 1
        )
        return [row async for row in cursor]

    async def get_quota(self, user_id: str, day: str) -> int | None:
        if not self._ready():
            return None
        doc = await self.db.claude_quotas.find_one({"_id": f"{user_id}:{day}"})
        return int(doc.get("count", 0)) if doc else 0

    async def try_increment_quota(self, user_id: str, day: str, cap: int) -> int | None:
        if not self._ready():
            return None
        key = f"{user_id}:{day}"
        now = _now()
        try:
            await self.db.claude_quotas.insert_one(
                {
                    "_id": key,
                    "user_id": user_id,
                    "day": day,
                    "count": 1,
                    "updated_at": now,
                }
            )
            return 1
        except DuplicateKeyError:
            pass
        doc = await self.db.claude_quotas.find_one_and_update(
            {"_id": key, "count": {"$lt": cap}},
            {"$inc": {"count": 1}, "$set": {"updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["count"]) if doc else None

    async def refund_quota(self, user_id: str, day: str) -> None:
        if not self._ready():
            return
        await self.db.claude_quotas.update_one(
            {"_id": f"{user_id}:{day}", "count": {"$gt": 0}},
            {"$inc": {"count": -1}, "$set": {"updated_at": _now()}},
        )

    async def mark_pro_seen(
        self, user_id: str, *, is_pro: bool
    ) -> dict[str, Any] | None:
        """Track per-user Pro state and detect free→Pro conversions.

        Returns ``{"converted": True, "first_seen_at": ..."}`` exactly once
        per user — the first time we see them as Pro after having recorded
        them as non-Pro at least once. Otherwise returns ``None``.

        Storing ``ever_non_pro`` lets us distinguish "user joined as Pro"
        (no conversion) from "user upgraded" (conversion). The atomic
        ``find_one_and_update`` on a guarded filter makes the conversion
        emit at-most-once even under concurrent requests.
        """
        if not self._ready() or not user_id:
            return None
        now = _now()
        set_fields: dict[str, Any] = {"last_seen_at": now, "is_pro": bool(is_pro)}
        if not is_pro:
            set_fields["ever_non_pro"] = True
        try:
            await self.db.pro_users.update_one(
                {"_id": user_id},
                {
                    "$setOnInsert": {"_id": user_id, "first_seen_at": now},
                    "$set": set_fields,
                },
                upsert=True,
            )
        except PyMongoError as e:
            logger.debug("mark_pro_seen upsert failed for %s: %s", user_id, e)
            return None

        if not is_pro:
            return None

        try:
            doc = await self.db.pro_users.find_one_and_update(
                {
                    "_id": user_id,
                    "ever_non_pro": True,
                    "first_seen_pro_at": {"$exists": False},
                },
                {"$set": {"first_seen_pro_at": now}},
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as e:
            logger.debug("mark_pro_seen conversion check failed for %s: %s", user_id, e)
            return None

        if not doc:
            return None
        return {
            "converted": True,
            "first_seen_at": (doc.get("first_seen_at") or now).isoformat(),
        }

    def _response_query(
        self,
        *,
        user_id: str,
        platform: str | None = None,
        progress: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if user_id != "dev":
            query["user_id"] = user_id

        def contains(field: str, value: str | None) -> None:
            if value:
                query[field] = {"$regex": re.escape(value), "$options": "i"}

        contains("platform", platform)
        contains("progress", progress)
        contains("model_name", model)
        contains("session_id", session_id)
        contains("job_id", job_id)
        if q:
            regex = {"$regex": re.escape(q), "$options": "i"}
            query["$or"] = [
                {"id": regex},
                {"session_id": regex},
                {"short_session_id": regex},
                {"session_title": regex},
                {"model_name": regex},
                {"platform": regex},
                {"run_type": regex},
                {"result_storage": regex},
                {"progress": regex},
                {"job_id": regex},
                {"final_artifact_or_result": regex},
                {"error": regex},
            ]
        return query

    def _mongo_safe_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            for key, item in value.items():
                text_key = str(key).replace(".", "_")
                if text_key.startswith("$"):
                    text_key = f"_{text_key[1:]}"
                safe[text_key] = self._mongo_safe_value(item)
            return safe
        if isinstance(value, list):
            return [self._mongo_safe_value(item) for item in value]
        return value

    def _normalize_response_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        progress = str(normalized.get("progress") or "").lower()
        state = str(
            (normalized.get("provider_metadata") or {}).get("state") or ""
        ).lower()
        status = state if progress in {"", "unknown"} else progress
        if status in {"succeeded", "success", "complete"}:
            normalized["progress"] = "completed"
        elif status == "failure":
            normalized["progress"] = "failed"
        elif status == "canceled":
            normalized["progress"] = "cancelled"
        elif status in {"scheduling", "scheduled", "pending"}:
            normalized["progress"] = "queued"
        elif status in {"billing_required", "missing_token", "approval_required"}:
            normalized["progress"] = "blocked"
        return normalized

    async def upsert_response_rows(
        self, rows: list[dict[str, Any]], *, user_id: str = "dev"
    ) -> None:
        if not self._ready() or not rows:
            return
        now = _now()
        ops: list[Any] = []
        raw_rows = [dict(raw) for raw in rows]
        row_ids = [str(row.get("id") or "") for row in raw_rows if row.get("id")]
        identity_filters = []
        for row in raw_rows:
            row_user_id = str(row.get("user_id") or user_id)
            platform = str(row.get("platform") or "")
            job_id = str(row.get("job_id") or "")
            session_id = str(row.get("session_id") or "")
            if platform and job_id:
                identity_filters.append(
                    {"user_id": row_user_id, "platform": platform, "job_id": job_id}
                )
                if session_id:
                    identity_filters.append(
                        {
                            "session_id": session_id,
                            "platform": platform,
                            "job_id": job_id,
                        }
                    )
                if platform == "hf-jobs":
                    slug_regex = _hf_job_slug_regex(job_id)
                    if slug_regex:
                        identity_filters.append(
                            {
                                "user_id": row_user_id,
                                "platform": platform,
                                "job_id": slug_regex,
                            }
                        )
                        if session_id:
                            identity_filters.append(
                                {
                                    "session_id": session_id,
                                    "platform": platform,
                                    "job_id": slug_regex,
                                }
                            )
        existing_by_id: dict[str, dict[str, Any]] = {}
        existing_candidates: list[dict[str, Any]] = []
        if row_ids:
            cursor = self.db.response_rows.find(
                {"_id": {"$in": row_ids}},
                {
                    "_id": 1,
                    "id": 1,
                    "user_id": 1,
                    "session_id": 1,
                    "platform": 1,
                    "job_id": 1,
                    "actual_sequence_number": 1,
                    "display_session_number": 1,
                    "batch_number": 1,
                    "created_at": 1,
                },
            )
            existing_by_id = {str(doc["_id"]): doc async for doc in cursor}
            existing_candidates.extend(existing_by_id.values())
        if identity_filters:
            cursor = self.db.response_rows.find(
                {"$or": identity_filters},
                {
                    "_id": 1,
                    "id": 1,
                    "user_id": 1,
                    "session_id": 1,
                    "platform": 1,
                    "job_id": 1,
                    "actual_sequence_number": 1,
                    "display_session_number": 1,
                    "batch_number": 1,
                    "created_at": 1,
                },
            )
            async for doc in cursor:
                existing_candidates.append(doc)
        for raw in raw_rows:
            row = self._mongo_safe_value(dict(raw))
            row_id = str(row.get("id") or "")
            if not row_id:
                continue
            row_user_id = str(row.get("user_id") or user_id)
            existing = existing_by_id.get(row_id) or _find_existing_response_row(
                existing_candidates, row, row_user_id
            )
            row.pop("_id", None)
            row.pop("inserted_at", None)
            row.pop("schema_version", None)
            for key in (
                "actual_sequence_number",
                "display_session_number",
                "batch_number",
                "created_at",
            ):
                if existing.get(key) is not None:
                    row[key] = existing[key]
            if existing.get("_id"):
                row_id = str(existing["_id"])
                row["id"] = str(existing.get("id") or existing["_id"])
            if (
                existing.get("job_id")
                and str(row.get("platform") or "") == "hf-jobs"
                and _same_hf_job_id(existing.get("job_id"), row.get("job_id"))
                and str(existing.get("job_id")).startswith("https://")
                and not str(row.get("job_id") or "").startswith("https://")
            ):
                row["job_id"] = existing["job_id"]
            row["user_id"] = row_user_id
            row["updated_at"] = now
            ops.append(
                UpdateOne(
                    {"_id": row_id},
                    {
                        "$set": row,
                        "$setOnInsert": {
                            "_id": row_id,
                            "inserted_at": now,
                            "schema_version": SCHEMA_VERSION,
                        },
                    },
                    upsert=True,
                )
            )
        if not ops:
            return
        try:
            await self.db.response_rows.bulk_write(ops, ordered=False)
        except PyMongoError as e:
            logger.debug("Failed to upsert response rows: %s", e)

    async def list_response_rows(
        self,
        *,
        user_id: str,
        page: int = 1,
        page_size: int = 50,
        platform: str | None = None,
        progress: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        safe_page = max(1, int(page or 1))
        safe_page_size = min(200, max(1, int(page_size or 50)))
        if not self._ready():
            return {
                "rows": [],
                "page": safe_page,
                "page_size": safe_page_size,
                "total_rows": 0,
                "total_pages": 0,
                "has_next": False,
                "has_previous": False,
            }
        query = self._response_query(
            user_id=user_id,
            platform=platform,
            progress=progress,
            model=model,
            session_id=session_id,
            job_id=job_id,
            q=q,
        )
        total_rows = await self.db.response_rows.count_documents(query)
        total_pages = math.ceil(total_rows / safe_page_size) if total_rows else 0
        cursor = (
            self.db.response_rows.find(query, {"_id": 0})
            .sort([("actual_sequence_number", -1), ("created_at", -1)])
            .skip((safe_page - 1) * safe_page_size)
            .limit(safe_page_size)
        )
        rows = [self._normalize_response_row(row) async for row in cursor]
        return {
            "rows": rows,
            "page": safe_page,
            "page_size": safe_page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "has_next": bool(total_pages and safe_page < total_pages),
            "has_previous": safe_page > 1 and total_pages > 0,
        }

    async def get_response_summary(self, *, user_id: str) -> dict[str, Any]:
        if not self._ready():
            return await super().get_response_summary(user_id=user_id)
        query: dict[str, Any] = {}
        if user_id != "dev":
            query["user_id"] = user_id
        total = await self.db.response_rows.count_documents(query)
        latest = await self.db.response_rows.find_one(
            query,
            sort=[("actual_sequence_number", -1), ("created_at", -1)],
        )
        batch = int((latest or {}).get("batch_number") or 1)
        return {
            "total_responses": total,
            "visible_count": min(total, 50),
            "batch_number": batch,
            "has_rows": bool(total),
            "button_enabled": True,
            "durable": True,
            "store_type": "mongodb",
        }


_store: NoopSessionStore | MongoSessionStore | None = None


def get_session_store() -> NoopSessionStore | MongoSessionStore:
    global _store
    if _store is None:
        uri = os.environ.get("MONGODB_URI")
        db_name = os.environ.get("MONGODB_DB", "ml-intern")
        _store = MongoSessionStore(uri, db_name) if uri else NoopSessionStore()
    return _store


def session_store_status(
    store: NoopSessionStore | MongoSessionStore | None = None,
) -> dict[str, Any]:
    """Return a non-secret health summary for hosted session durability."""
    active_store = store or get_session_store()
    if isinstance(active_store, MongoSessionStore) or getattr(
        active_store, "enabled", False
    ):
        durable = bool(active_store.enabled)
        return {
            "type": "mongodb",
            "durable": durable,
            "warning": None
            if durable
            else "MongoDB session persistence is not available; sessions will not survive restarts",
        }
    return {
        "type": "noop",
        "durable": False,
        "warning": NO_DURABLE_STORE_WARNING,
    }


def _reset_store_for_tests(
    store: NoopSessionStore | MongoSessionStore | None = None,
) -> None:
    global _store
    _store = store
