"""Read-only dataset discovery planning helpers.

This module normalizes candidate metadata produced by other search/research
tools. It does not crawl the web, call external APIs, download data, or launch
training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_ALLOWED_SOURCES = ["huggingface", "github", "papers", "public_web"]
DEFAULT_EXCLUDED_SOURCES = ["kaggle"]

SOURCE_LABELS = {
    "huggingface": "Hugging Face Datasets",
    "github": "GitHub",
    "papers": "papers",
    "public_web": "public web",
    "kaggle": "Kaggle",
}


@dataclass(frozen=True)
class DatasetCandidate:
    name: str
    source: str
    url: str | None
    domain: str
    task_type: str
    license: str | None
    size: str | None
    schema_hint: list[str]
    quality_notes: list[str]
    risks: list[str]
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetDiscoveryPlan:
    domain: str
    task_type: str
    allowed_sources: list[str]
    excluded_sources: list[str]
    candidates: list[DatasetCandidate]
    recommendation: str
    requires_user_selection: bool = True
    provider: str = "hf-jobs"
    user_goal: str | None = None
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_source(source: str | None) -> str:
    return (source or "huggingface").strip().lower().replace("-", "_")


def _normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source.replace("_", " ").title())


def normalize_candidate(
    candidate: DatasetCandidate | dict[str, Any],
    *,
    domain: str,
    task_type: str,
) -> DatasetCandidate:
    if isinstance(candidate, DatasetCandidate):
        return candidate

    score = candidate.get("score", 0.0)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = 0.0
    score = max(0.0, min(float(score), 1.0))

    return DatasetCandidate(
        name=str(candidate.get("name") or "Unnamed dataset"),
        source=_normalize_source(candidate.get("source")),
        url=str(candidate["url"]) if candidate.get("url") else None,
        domain=str(candidate.get("domain") or domain),
        task_type=str(candidate.get("task_type") or task_type),
        license=str(candidate["license"]) if candidate.get("license") else None,
        size=str(candidate["size"]) if candidate.get("size") else None,
        schema_hint=_normalize_list(candidate.get("schema_hint")),
        quality_notes=_normalize_list(candidate.get("quality_notes")),
        risks=_normalize_list(candidate.get("risks")),
        score=score,
        reason=str(candidate.get("reason") or "Candidate supplied by discovery."),
    )


def rank_candidates(candidates: list[DatasetCandidate]) -> list[DatasetCandidate]:
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _normalize_sources(value: list[str] | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    normalized = []
    for source in value:
        item = _normalize_source(source)
        if item and item not in normalized:
            normalized.append(item)
    return normalized or list(default)


def _default_next_steps() -> list[str]:
    return [
        "search Hugging Face Datasets for matching public datasets",
        "search papers/research for benchmark datasets and data recipes",
        "search GitHub for public dataset repos or conversion scripts",
        "inspect candidate schemas, licenses, privacy notes, and quality risks",
        "ask the user to approve/select one dataset before training",
    ]


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
    normalized_domain = (domain or "general").strip().lower().replace("-", "_")
    normalized_task = (task_type or "general").strip().lower().replace("-", "_")
    allowed = _normalize_sources(allowed_sources, DEFAULT_ALLOWED_SOURCES)
    excluded = _normalize_sources(excluded_sources, DEFAULT_EXCLUDED_SOURCES)

    # Kaggle is explicitly not connected in this phase, even if a caller passes
    # it by mistake.
    allowed = [source for source in allowed if source != "kaggle"]
    if "kaggle" not in excluded:
        excluded.append("kaggle")

    normalized_candidates = [
        normalize_candidate(
            candidate, domain=normalized_domain, task_type=normalized_task
        )
        for candidate in (candidates or [])
    ]

    return DatasetDiscoveryPlan(
        domain=normalized_domain,
        task_type=normalized_task,
        allowed_sources=allowed,
        excluded_sources=excluded,
        candidates=rank_candidates(normalized_candidates),
        recommendation=(
            "No uploaded dataset detected. Research allowed public sources, inspect "
            "schema/license/privacy/quality for each candidate, then ask the user "
            "to approve a selected dataset before any training plan or job launch."
        ),
        requires_user_selection=True,
        provider=(provider or "hf-jobs").strip().lower(),
        user_goal=user_goal,
        next_steps=_default_next_steps(),
    )


def format_dataset_discovery_plan(plan: DatasetDiscoveryPlan) -> str:
    lines = [
        "## Dataset Discovery Plan",
        "",
        "No uploaded dataset detected.",
        f"**Provider:** {plan.provider}",
        f"**Domain:** {plan.domain}",
        f"**Task type:** {plan.task_type}",
    ]
    if plan.user_goal:
        lines.append(f"**User goal:** {plan.user_goal}")

    lines.extend(
        [
            "",
            "### Allowed Sources",
            *[f"- {_source_label(source)}" for source in plan.allowed_sources],
            "",
            "### Excluded Sources",
            *[
                f"- {_source_label(source)} (not connected in this version; future work only)"
                if source == "kaggle"
                else f"- {_source_label(source)}"
                for source in plan.excluded_sources
            ],
            "",
            "### Recommended Next Steps",
            *[f"- {step}" for step in plan.next_steps],
            "",
            "### Candidate Ranking",
        ]
    )

    if not plan.candidates:
        lines.append(
            "- No candidates supplied yet. Use the allowed search/research tools, "
            "then pass discovered candidates back for normalization and ranking."
        )
    else:
        for index, candidate in enumerate(plan.candidates, start=1):
            lines.extend(
                [
                    f"{index}. **{candidate.name}** ({_source_label(candidate.source)}, score {candidate.score:.2f})",
                    f"   - Reason: {candidate.reason}",
                    f"   - URL: {candidate.url or 'Not provided'}",
                    f"   - License: {candidate.license or 'Unknown; verify before use'}",
                    f"   - Size: {candidate.size or 'Unknown'}",
                    f"   - Schema hint: {', '.join(candidate.schema_hint) if candidate.schema_hint else 'Unknown; inspect before training'}",
                    f"   - Quality notes: {', '.join(candidate.quality_notes) if candidate.quality_notes else 'None supplied'}",
                    f"   - Risks: {', '.join(candidate.risks) if candidate.risks else 'Verify license, privacy, and fit before training'}",
                ]
            )

    lines.extend(
        [
            "",
            "### Recommendation",
            plan.recommendation,
            "",
            "Please select or approve one dataset before training. Do not launch "
            "cloud training until the selected dataset schema and license are inspected.",
            "",
            "Planning only: this helper never crawls sources, downloads datasets, "
            "launches jobs, makes cloud calls, uploads data, or spends money.",
        ]
    )
    return "\n".join(lines)
