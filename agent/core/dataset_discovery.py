"""Read-only dataset discovery planning helpers.

This module turns metadata from existing safe search/research tools into a
ranked recommendation. It does not crawl the web, call external APIs, download
data, or launch training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import re
from typing import Any

from agent.core.redact import redact_json_like, redact_text


DEFAULT_ALLOWED_SOURCES = ["huggingface", "github", "papers", "public_web"]
DEFAULT_EXCLUDED_SOURCES = ["kaggle"]
DIRECTLY_LOADABLE_SOURCES = {"huggingface", "session_upload"}

SOURCE_LABELS = {
    "huggingface": "Hugging Face Datasets",
    "github": "GitHub",
    "papers": "papers",
    "public_web": "public web",
    "session_upload": "User-uploaded dataset metadata",
    "kaggle": "Kaggle",
}

CLEAR_LICENSES = {
    "mit",
    "apache-2.0",
    "apache 2.0",
    "bsd-3-clause",
    "bsd",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "openrail",
    "odc-by",
}
RESTRICTIVE_LICENSE_TOKENS = {"non-commercial", "nc", "research only", "gpl"}
TEXT_COLUMNS = {
    "messages",
    "text",
    "prompt",
    "completion",
    "instruction",
    "output",
    "response",
    "question",
    "answer",
    "chosen",
    "rejected",
}
LABEL_COLUMNS = {"label", "labels", "target", "class", "category", "price", "score"}
HIGH_PRIVACY_TERMS = {
    "medical",
    "patient",
    "health",
    "diagnosis",
    "symptom",
    "personal",
    "private",
    "pii",
    "ssn",
    "email",
    "phone",
    "address",
}
COMPLIANCE_TERMS = {
    "finance",
    "financial",
    "legal",
    "loan",
    "credit",
    "bank",
    "contract",
}
SHORT_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{6,}|[A-Za-z0-9_-]*secret[A-Za-z0-9_-]*)\b",
    re.I,
)


@dataclass(frozen=True)
class DatasetIntent:
    query: str
    domain: str = "general"
    task_type: str = "sft"
    target_provider: str = "hf-jobs"
    model_preference: str | None = None
    uploaded_vs_no_upload_intent: str = "no_upload"
    data_modality: str = "text"
    privacy_sensitivity: str = "unknown"
    license_sensitivity: str = "standard"
    expected_size: str = "unknown"
    columns_needed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetRisk:
    category: str
    severity: str
    message: str
    status: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSchemaSummary:
    status: str = "unknown"
    columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetLicenseSummary:
    license: str | None = None
    status: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetQualityScore:
    relevance_score: float = 0.0
    safety_score: float = 0.0
    license_score: float = 0.0
    schema_score: float = 0.0
    size_score: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetCandidate:
    dataset_id: str = "unnamed-dataset"
    source: str = "huggingface"
    source_url: str | None = None
    repo_id: str | None = None
    config: str | None = None
    split: str | None = None
    title: str = "Unnamed dataset"
    description: str | None = None
    domain: str = "general"
    task_type: str = "sft"
    license: str | None = None
    license_status: str = "unknown"
    privacy_status: str = "unknown"
    schema_status: str = "unknown"
    quality_score: DatasetQualityScore = field(default_factory=DatasetQualityScore)
    relevance_score: float = 0.0
    safety_score: float = 0.0
    license_score: float = 0.0
    schema_score: float = 0.0
    size_score: float = 0.0
    overall_score: float = 0.0
    row_count: int | None = None
    columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    recommended_use: str | None = None
    reasons: list[str] = field(default_factory=list)
    risks: list[DatasetRisk | str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    excluded: bool = False
    exclusion_reason: str | None = None
    load_dataset_snippet: str | None = None
    # Legacy aliases retained for existing planner tests/tool callers.
    name: str | None = None
    url: str | None = None
    size: str | None = None
    schema_hint: list[str] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)
    score: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        title = self.title or self.name or self.dataset_id or "Unnamed dataset"
        source_url = self.source_url or self.url
        overall = self.overall_score if self.score is None else _clamp(self.score)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "name", self.name or title)
        object.__setattr__(self, "source_url", source_url)
        object.__setattr__(self, "url", self.url or source_url)
        object.__setattr__(self, "score", overall)
        object.__setattr__(self, "overall_score", overall)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risks"] = [
            risk.to_dict() if isinstance(risk, DatasetRisk) else risk
            for risk in self.risks
        ]
        return redact_json_like(payload)


@dataclass(frozen=True)
class DatasetDiscoveryResult:
    query: str
    intent: DatasetIntent
    allowed_sources: list[str]
    excluded_sources: list[str]
    candidates: list[DatasetCandidate]
    warnings: list[str] = field(default_factory=list)
    selected_candidate: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    requires_user_selection: bool = True

    @property
    def recommended_candidate(self) -> DatasetCandidate | None:
        return next(
            (candidate for candidate in self.candidates if not candidate.excluded), None
        )

    @property
    def domain(self) -> str:
        return self.intent.domain

    @property
    def task_type(self) -> str:
        return self.intent.task_type

    @property
    def provider(self) -> str:
        return self.intent.target_provider

    @property
    def recommendation(self) -> str:
        if self.recommended_candidate:
            return (
                "Recommended candidate found, but user selection and approval are "
                "required before any training launch."
            )
        return "Dataset selection required before training."

    @property
    def next_steps(self) -> list[str]:
        return _default_next_steps()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        recommended = self.recommended_candidate
        payload["recommended_candidate"] = (
            recommended.to_dict() if recommended else None
        )
        return redact_json_like(payload)


DatasetDiscoveryPlan = DatasetDiscoveryResult


def _clamp(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    return default


def _text(value: Any, default: str = "") -> str:
    if value in (None, ""):
        return default
    return SHORT_SECRET_RE.sub("[REDACTED]", redact_text(str(value).strip()))


def _normalize_source(source: Any) -> str:
    return _text(source, "huggingface").lower().replace("-", "_")


def _normalize_domain(value: Any) -> str:
    return _text(value, "general").lower().replace("-", "_").replace(" ", "_")


def _normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [redact_text(str(item).strip()) for item in value if str(item).strip()]


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source.replace("_", " ").title())


def _normalize_sources(value: list[str] | None, default: list[str]) -> list[str]:
    raw = value or default
    normalized: list[str] = []
    for source in raw:
        item = _normalize_source(source)
        if item and item not in normalized:
            normalized.append(item)
    return normalized or list(default)


def _provider_from_text(text: str, fallback: str = "hf-jobs") -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("gcloud", "gcp", "vertex", "google cloud")):
        return "gcp-vertex"
    if any(token in lowered for token in ("aws", "sagemaker", "sage maker")):
        return "aws-sagemaker"
    if any(
        token in lowered
        for token in ("hugging face", "huggingface", "hf jobs", "hf-jobs")
    ):
        return "hf-jobs"
    return fallback


def extract_dataset_intent(
    query: str | None, *, provider: str | None = None
) -> DatasetIntent:
    clean_query = _text(query, "")
    text = clean_query.lower()
    domain = "general"
    task_type = "sft"
    modality = "text"
    columns_needed = ["text"]
    privacy = "unknown"

    if any(token in text for token in HIGH_PRIVACY_TERMS):
        domain, privacy, columns_needed = (
            "medical",
            "high",
            ["question", "answer", "text"],
        )
    elif "manufacturing" in text or "factory" in text:
        domain, modality, columns_needed = (
            "manufacturing",
            "mixed",
            ["instruction", "output", "text"],
        )
    elif "ipl" in text or "cricket" in text:
        domain, columns_needed = "sports_cricket_ipl", ["question", "answer", "text"]
    elif "hardware" in text or "troubleshooting" in text:
        domain, columns_needed = (
            "hardware_support",
            ["instruction", "output", "question", "answer"],
        )
    elif "house price" in text or "real estate" in text or "property" in text:
        domain, task_type, modality = "real_estate", "regression", "tabular"
        columns_needed = ["price", "features", "text"]
    elif any(token in text for token in COMPLIANCE_TERMS):
        domain, privacy = (
            "finance" if "finance" in text or "credit" in text else "legal",
            "high",
        )

    if privacy == "unknown" and any(
        token in text for token in HIGH_PRIVACY_TERMS | COMPLIANCE_TERMS
    ):
        privacy = "high"
    elif privacy == "unknown" and domain in {"medical", "finance", "legal"}:
        privacy = "high"

    if any(token in text for token in ("classification", "classify", "label")):
        task_type = "classification"
    if (
        any(token in text for token in ("summarization", "summarize", "qa", "facts"))
        and task_type == "sft"
    ):
        task_type = "sft"

    expected_size = "small" if "small" in text or "demo" in text else "unknown"
    uploaded_intent = (
        "uploaded"
        if any(token in text for token in ("uploaded", "my dataset", "attached"))
        else "no_upload"
    )
    license_sensitivity = (
        "high"
        if any(token in text for token in ("commercial", "license", "legal"))
        else "standard"
    )

    return DatasetIntent(
        query=clean_query,
        domain=domain,
        task_type=task_type,
        target_provider=_provider_from_text(text, provider or "hf-jobs"),
        uploaded_vs_no_upload_intent=uploaded_intent,
        data_modality=modality,
        privacy_sensitivity=privacy,
        license_sensitivity=license_sensitivity,
        expected_size=expected_size,
        columns_needed=columns_needed,
    )


def _license_summary(license_value: str | None) -> DatasetLicenseSummary:
    license_text = _text(license_value) or None
    if not license_text:
        return DatasetLicenseSummary(
            None, "missing", ["Missing license; verify before use."]
        )
    lowered = license_text.lower()
    if any(token in lowered for token in RESTRICTIVE_LICENSE_TOKENS):
        return DatasetLicenseSummary(
            license_text, "restrictive", ["License may restrict commercial training."]
        )
    if lowered in CLEAR_LICENSES or any(clear in lowered for clear in CLEAR_LICENSES):
        return DatasetLicenseSummary(
            license_text, "clear", ["License is machine-readable; still verify fit."]
        )
    if lowered in {"unknown", "other", "n/a"}:
        return DatasetLicenseSummary(
            license_text,
            "unknown",
            ["License is unknown; do not train without confirmation."],
        )
    return DatasetLicenseSummary(
        license_text, "unclear", ["License needs manual review."]
    )


def _schema_summary(columns: list[str], task_type: str) -> DatasetSchemaSummary:
    normalized = [column.strip() for column in columns if column.strip()]
    lowered = {column.lower() for column in normalized}
    text_columns = [column for column in normalized if column.lower() in TEXT_COLUMNS]
    label_columns = [column for column in normalized if column.lower() in LABEL_COLUMNS]
    notes: list[str] = []
    if not normalized:
        return DatasetSchemaSummary("unknown", [], [], [], ["No columns supplied."])
    if task_type == "regression":
        status = "compatible" if label_columns else "needs_mapping"
    elif (
        {"messages"} & lowered
        or {"prompt", "completion"} <= lowered
        or {"instruction", "output"} <= lowered
        or {"question", "answer"} <= lowered
    ):
        status = "compatible"
    elif text_columns:
        status = "needs_mapping"
        notes.append("Text-like columns found but training mapping must be confirmed.")
    else:
        status = "unsupported"
        notes.append("No text/instruction columns found.")
    return DatasetSchemaSummary(status, normalized, text_columns, label_columns, notes)


def _privacy_status(
    intent: DatasetIntent, text_blob: str
) -> tuple[str, list[DatasetRisk]]:
    lowered = text_blob.lower()
    risks: list[DatasetRisk] = []
    if intent.privacy_sensitivity == "high" or any(
        token in lowered for token in HIGH_PRIVACY_TERMS
    ):
        risks.append(
            DatasetRisk(
                "privacy",
                "warning",
                "Medical/personal data may contain private or regulated records.",
            )
        )
        return "high", risks
    if any(token in lowered for token in COMPLIANCE_TERMS):
        risks.append(
            DatasetRisk(
                "compliance",
                "warning",
                "Finance/legal data requires compliance review.",
            )
        )
        return "medium", risks
    return "low", risks


def _row_count(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _score_candidate(
    *,
    intent: DatasetIntent,
    source: str,
    title: str,
    description: str | None,
    license_status: str,
    privacy_status: str,
    schema_status: str,
    row_count: int | None,
    excluded: bool,
) -> DatasetQualityScore:
    text = f"{title} {description or ''}".lower()
    domain_tokens = intent.domain.replace("_", " ").split()
    relevance = 0.45 + min(0.35, 0.12 * sum(token in text for token in domain_tokens))
    if source == "huggingface":
        relevance += 0.08
    if excluded:
        return DatasetQualityScore(overall_score=0.0)
    license_score = {
        "clear": 1.0,
        "unclear": 0.45,
        "restrictive": 0.25,
        "missing": 0.15,
        "unknown": 0.25,
    }.get(license_status, 0.25)
    safety_score = {"low": 0.95, "medium": 0.55, "high": 0.2, "unknown": 0.45}.get(
        privacy_status, 0.45
    )
    schema_score = {
        "compatible": 1.0,
        "needs_mapping": 0.6,
        "unknown": 0.35,
        "unsupported": 0.05,
    }.get(schema_status, 0.35)
    if row_count is None:
        size_score = 0.45
    elif row_count < 50:
        size_score = 0.2
    elif row_count > 1_000_000:
        size_score = 0.45
    elif row_count > 100_000:
        size_score = 0.7
    else:
        size_score = 0.9
    overall = (
        relevance * 0.32
        + license_score * 0.18
        + safety_score * 0.2
        + schema_score * 0.2
        + size_score * 0.1
    )
    return DatasetQualityScore(
        relevance_score=_clamp(relevance),
        safety_score=_clamp(safety_score),
        license_score=_clamp(license_score),
        schema_score=_clamp(schema_score),
        size_score=_clamp(size_score),
        overall_score=_clamp(overall),
    )


def _snippet(
    source: str, repo_id: str | None, config: str | None, split: str | None
) -> str | None:
    if source != "huggingface" or not repo_id:
        return None
    args = [repr(repo_id)]
    if config:
        args.append(repr(config))
    kwargs = f", split={split!r}" if split else ""
    return f"from datasets import load_dataset\n\ndataset = load_dataset({', '.join(args)}{kwargs})"


def normalize_candidate(
    candidate: DatasetCandidate | dict[str, Any],
    *,
    intent: DatasetIntent | None = None,
    domain: str | None = None,
    task_type: str | None = None,
) -> DatasetCandidate:
    if isinstance(candidate, DatasetCandidate):
        return candidate
    intent = intent or DatasetIntent(
        query="",
        domain=_normalize_domain(domain),
        task_type=_normalize_domain(task_type or "sft"),
    )
    source = _normalize_source(candidate.get("source"))
    dataset_id = _text(
        candidate.get("dataset_id")
        or candidate.get("repo_id")
        or candidate.get("name"),
        "unnamed-dataset",
    )
    title = _text(candidate.get("title") or candidate.get("name"), dataset_id)
    description = _text(candidate.get("description")) or None
    columns = _normalize_list(candidate.get("columns") or candidate.get("schema_hint"))
    row_count = _row_count(candidate.get("row_count") or candidate.get("rows"))
    license_summary = _license_summary(candidate.get("license"))
    schema_summary = _schema_summary(
        columns, str(candidate.get("task_type") or intent.task_type)
    )
    text_blob = " ".join([title, description or "", dataset_id, " ".join(columns)])
    privacy_status, privacy_risks = _privacy_status(intent, text_blob)
    excluded = source == "kaggle"
    exclusion_reason = "Kaggle is future work only." if excluded else None
    warnings = _normalize_list(candidate.get("warnings"))
    risks: list[DatasetRisk | str] = [*privacy_risks]
    risks.extend(_normalize_list(candidate.get("risks")))
    if license_summary.status != "clear":
        warnings.extend(license_summary.notes)
    if schema_summary.status != "compatible":
        warnings.extend(schema_summary.notes)
    if row_count is not None and row_count < 50:
        warnings.append("Dataset may be too small for useful fine-tuning.")
    if row_count is not None and row_count > 1_000_000:
        warnings.append("Dataset may be too large; use capped pilot samples first.")
    if excluded and exclusion_reason:
        warnings.append(exclusion_reason)
    quality = _score_candidate(
        intent=intent,
        source=source,
        title=title,
        description=description,
        license_status=license_summary.status,
        privacy_status=privacy_status,
        schema_status=schema_summary.status,
        row_count=row_count,
        excluded=excluded,
    )
    if isinstance(candidate.get("score"), int | float):
        quality = DatasetQualityScore(
            relevance_score=quality.relevance_score,
            safety_score=quality.safety_score,
            license_score=quality.license_score,
            schema_score=quality.schema_score,
            size_score=quality.size_score,
            overall_score=_clamp(candidate.get("score")),
        )
    reasons = _normalize_list(candidate.get("reasons"))
    if candidate.get("reason"):
        reasons.append(_text(candidate.get("reason")))
    if not reasons:
        reasons.append("Candidate supplied by discovery metadata.")
    repo_id = _text(candidate.get("repo_id")) or (
        dataset_id if source == "huggingface" and "/" in dataset_id else None
    )
    return DatasetCandidate(
        dataset_id=dataset_id,
        source=source,
        source_url=_text(candidate.get("source_url") or candidate.get("url")) or None,
        repo_id=repo_id,
        config=_text(candidate.get("config")) or None,
        split=_text(candidate.get("split")) or None,
        title=title,
        description=description,
        domain=_normalize_domain(candidate.get("domain") or intent.domain),
        task_type=_normalize_domain(candidate.get("task_type") or intent.task_type),
        license=license_summary.license,
        license_status=license_summary.status,
        privacy_status=privacy_status,
        schema_status=schema_summary.status,
        quality_score=quality,
        relevance_score=quality.relevance_score,
        safety_score=quality.safety_score,
        license_score=quality.license_score,
        schema_score=quality.schema_score,
        size_score=quality.size_score,
        overall_score=quality.overall_score,
        row_count=row_count,
        columns=schema_summary.columns,
        text_columns=schema_summary.text_columns,
        label_columns=schema_summary.label_columns,
        recommended_use=(
            "Safe/loadable candidate to inspect and approve before training."
            if source in DIRECTLY_LOADABLE_SOURCES and not excluded
            else "Reference only unless a directly loadable dataset repo is confirmed."
        ),
        reasons=reasons,
        risks=risks,
        warnings=list(dict.fromkeys(warnings)),
        excluded=excluded,
        exclusion_reason=exclusion_reason,
        load_dataset_snippet=_snippet(
            source,
            repo_id,
            _text(candidate.get("config")) or None,
            _text(candidate.get("split")) or None,
        ),
        name=title,
        url=_text(candidate.get("url") or candidate.get("source_url")) or None,
        size=_text(candidate.get("size"))
        or (f"{row_count} rows" if row_count is not None else None),
        schema_hint=schema_summary.columns,
        quality_notes=_normalize_list(candidate.get("quality_notes")),
        score=quality.overall_score,
        reason=reasons[0],
    )


def rank_candidates(candidates: list[DatasetCandidate]) -> list[DatasetCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.excluded,
            -candidate.overall_score,
            candidate.title,
        ),
    )


def _default_next_steps() -> list[str]:
    return [
        "search Hugging Face Datasets for matching public datasets",
        "search papers/research for benchmark datasets and data recipes",
        "search GitHub for public dataset repos or conversion scripts",
        "inspect candidate schemas, licenses, privacy notes, and quality risks",
        "ask the user to approve/select one dataset before training",
    ]


def build_dataset_discovery_result(
    *,
    query: str | None = None,
    domain: str | None = None,
    task_type: str | None = None,
    user_goal: str | None = None,
    provider: str = "hf-jobs",
    allowed_sources: list[str] | None = None,
    excluded_sources: list[str] | None = None,
    candidates: list[DatasetCandidate | dict[str, Any]] | None = None,
    selected_candidate: dict[str, Any] | None = None,
) -> DatasetDiscoveryResult:
    intent = extract_dataset_intent(query or user_goal or "", provider=provider)
    if domain:
        intent = DatasetIntent(
            **{**intent.to_dict(), "domain": _normalize_domain(domain)}
        )
    if task_type:
        intent = DatasetIntent(
            **{**intent.to_dict(), "task_type": _normalize_domain(task_type)}
        )
    allowed = [
        source
        for source in _normalize_sources(allowed_sources, DEFAULT_ALLOWED_SOURCES)
        if source != "kaggle"
    ]
    excluded = _normalize_sources(excluded_sources, DEFAULT_EXCLUDED_SOURCES)
    if "kaggle" not in excluded:
        excluded.append("kaggle")
    normalized = [
        normalize_candidate(candidate, intent=intent)
        for candidate in (candidates or [])
    ]
    ranked = rank_candidates(normalized)
    warnings = ["User selection required before training."]
    if not ranked:
        warnings.append("No candidate datasets supplied yet.")
    if any(candidate.excluded for candidate in ranked):
        warnings.append("Kaggle is excluded as future work only.")
    return DatasetDiscoveryResult(
        query=redact_text(query or user_goal or ""),
        intent=intent,
        allowed_sources=allowed,
        excluded_sources=excluded,
        candidates=ranked,
        warnings=list(dict.fromkeys(warnings)),
        selected_candidate=redact_json_like(selected_candidate)
        if selected_candidate
        else None,
    )


def build_dataset_discovery_plan(
    *,
    domain: str,
    task_type: str,
    user_goal: str | None = None,
    provider: str = "hf-jobs",
    allowed_sources: list[str] | None = None,
    excluded_sources: list[str] | None = None,
    candidates: list[DatasetCandidate | dict[str, Any]] | None = None,
) -> DatasetDiscoveryPlan:
    return build_dataset_discovery_result(
        query=user_goal,
        domain=domain,
        task_type=task_type,
        user_goal=user_goal,
        provider=provider,
        allowed_sources=allowed_sources,
        excluded_sources=excluded_sources,
        candidates=candidates,
    )


def _candidate_lines(
    candidate: DatasetCandidate, index: int, recommended_id: str | None
) -> list[str]:
    badge = " Recommended" if candidate.dataset_id == recommended_id else ""
    if candidate.excluded:
        badge = " Excluded"
    lines = [
        f"{index}. **{candidate.title}**{badge} ({_source_label(candidate.source)}, score {candidate.overall_score:.2f})",
        f"   - Dataset ID: {candidate.dataset_id}",
        f"   - Overall score: {candidate.overall_score:.2f}; relevance {candidate.relevance_score:.2f}; license {candidate.license_score:.2f}; privacy/safety {candidate.safety_score:.2f}; schema {candidate.schema_score:.2f}",
        f"   - License: {candidate.license or 'Unknown'} ({candidate.license_status})",
        f"   - Privacy: {candidate.privacy_status}",
        f"   - Schema: {candidate.schema_status}",
        f"   - Rows: {candidate.row_count if candidate.row_count is not None else 'Unknown'}",
        f"   - Columns: {', '.join(candidate.columns) if candidate.columns else 'Unknown'}",
        f"   - Reasons: {'; '.join(candidate.reasons) if candidate.reasons else 'Candidate supplied by discovery.'}",
        f"   - Warnings: {'; '.join(candidate.warnings) if candidate.warnings else 'Confirm license, privacy, and schema before training.'}",
    ]
    if candidate.excluded and candidate.exclusion_reason:
        lines.append(f"   - Excluded: {candidate.exclusion_reason}")
    if candidate.load_dataset_snippet:
        lines.extend(
            [
                "   - load_dataset snippet:",
                "```python",
                candidate.load_dataset_snippet,
                "```",
            ]
        )
    return lines


def format_dataset_discovery_plan(plan: DatasetDiscoveryPlan) -> str:
    recommended = plan.recommended_candidate
    lines = [
        "## Dataset Discovery Plan",
        "",
        "No uploaded dataset detected.",
        "",
        "### Extracted Intent",
        f"- Provider: {plan.intent.target_provider}",
        f"- Domain: {plan.intent.domain}",
        f"- Task type: {plan.intent.task_type}",
        f"- Data modality: {plan.intent.data_modality}",
        f"- Privacy sensitivity: {plan.intent.privacy_sensitivity}",
        f"- Columns needed: {', '.join(plan.intent.columns_needed) if plan.intent.columns_needed else 'Unknown'}",
        "",
        "### Allowed Sources",
        *[f"- {_source_label(source)}" for source in plan.allowed_sources],
        "",
        "### Excluded Sources",
        *[
            "- Kaggle (future work only; not connected)"
            if source == "kaggle"
            else f"- {_source_label(source)}"
            for source in plan.excluded_sources
        ],
        "",
        "### Recommended Next Steps",
        *[f"- {step}" for step in _default_next_steps()],
        "",
        "### Candidate Ranking",
    ]
    if not plan.candidates:
        lines.append(
            "- No candidates supplied yet. Search allowed public sources, then inspect schema, license, privacy, and quality before training."
        )
    else:
        recommended_id = recommended.dataset_id if recommended else None
        for index, candidate in enumerate(plan.candidates, start=1):
            lines.extend(_candidate_lines(candidate, index, recommended_id))
    lines.extend(
        [
            "",
            "### Recommendation",
            (
                f"Recommended candidate: {recommended.title}. Please select and approve a dataset before training."
                if recommended
                else "No directly usable candidate selected. Please select and approve a dataset before training."
            ),
            "User selection required before training.",
            "Do not launch cloud training until the user approves one dataset and its schema/license/privacy fit.",
            "",
            "Planning only: this helper never crawls sources, downloads datasets, launch jobs, makes cloud calls, uploads data, or spends money.",
        ]
    )
    return "\n".join(lines)
