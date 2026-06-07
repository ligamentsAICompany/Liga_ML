# Dataset Discovery

Phase 6 improves the no-upload dataset path. When a user asks to train or
fine-tune without an uploaded dataset, Liga ML now creates a safe discovery
recommendation before planning or launch approval.

## Supported Sources

- Hugging Face Hub datasets, using metadata and dataset-card style fields only.
- Existing curated public docs, examples, papers, and GitHub references exposed
  through the current read-only tools.
- User-uploaded dataset metadata already attached to the session.

Kaggle is excluded in Phase 6. It is marked as future work only and is not
connected, queried, downloaded from, or treated as an available source.

Private or unverified sites, datasets that require manual credentials, unclear
licensing, and likely personal/private data are treated as risky or excluded.

## Intent Extraction

The discovery helper extracts:

- Domain and task type.
- Target provider: HF Jobs, Google Cloud Vertex AI, or AWS SageMaker.
- Uploaded-vs-no-upload intent.
- Data modality, privacy sensitivity, license sensitivity, expected size, and
  columns needed.

The extractor is deterministic and covered by fixtures for manufacturing, IPL
cricket, hardware troubleshooting, medical, and house price prediction prompts.

## Candidate Model

Each candidate can carry:

- Dataset identity: `dataset_id`, `source`, `source_url`, `repo_id`, `config`,
  `split`, `title`, and `description`.
- Fit metadata: `domain`, `task_type`, row count, columns, text columns, and
  label columns.
- Risk metadata: license, license status, privacy status, schema status, risks,
  warnings, exclusion state, and exclusion reason.
- Scores: relevance, safety/privacy, license, schema, size, and overall.
- Recommendation fields: reasons, recommended use, selected candidate, and a
  `load_dataset` snippet when a Hugging Face dataset repo is directly loadable.

## Scoring And Ranking

Ranking is deterministic and does not download datasets. It scores relevance to
the extracted intent, license clarity, privacy risk, schema compatibility,
row-count suitability, source compatibility, and quality hints. Excluded
candidates always rank after non-excluded candidates.

Research/docs/GitHub references are useful context but rank below directly
loadable Hugging Face or uploaded-dataset metadata unless a dataset repo is
confirmed.

## Risk Checks

License status values are `clear`, `unclear`, `restrictive`, `missing`, and
`unknown`.

Privacy status values are `low`, `medium`, `high`, and `unknown`.

Schema status values are `compatible`, `needs_mapping`, `unsupported`, and
`unknown`.

Warnings cover medical/personal data, finance/legal compliance, missing or
unclear licenses, unclear dataset cards, missing text/instruction columns, very
small datasets, very large datasets, and Kaggle exclusion.

## Planner And Approval Behavior

Discovery is a recommendation layer. It does not mean the dataset is already
available or selected. The planner can mention a recommended safe/loadable
candidate, explain risks and schema mapping notes, and still requires user
selection and approval before any cloud launch.

Uploaded datasets remain the preferred path when a normalized session upload is
available. The no-upload discovery tool tells the agent to use the uploaded
dataset first unless the user explicitly asks for alternatives.

Phase 7 consumes discovery metadata as planner context. Recommended candidates,
row-count hints, domain/sensitivity, schema risks, and license warnings can
influence model size, provider fallback, output policy, sample caps, and safety
warnings. Discovery remains advisory and does not download data or bypass launch
approval.

## Persistence And APIs

Structured discovery results are persisted in session/run state when emitted by
the `dataset_discovery` tool. Run summaries can include `dataset_discovery`, and
provider metadata can carry the same sanitized payload.

APIs:

- `GET /api/session/{session_id}/dataset-discovery`
- `GET /api/session/{session_id}/runs/{run_id}/dataset-discovery`

The response includes query, intent, allowed sources, excluded sources,
candidates, recommended candidate, warnings, selected candidate, timestamp, and
`requires_user_selection`.

## Audit And Usage

Audit timeline events include:

- `dataset_discovery_started`
- `dataset_candidates_found`
- `dataset_candidate_recommended`
- `dataset_candidate_excluded`
- `dataset_discovery_failed`

Discovery itself is read-only and should not add paid usage entries. Existing
LLM usage is still tracked by the normal run/session usage mechanisms.

## Frontend

The Dataset Discovery panel renders extracted intent, allowed/excluded sources,
Kaggle future-work status, candidate cards, scores, row counts, columns,
warnings, structured risks, snippets, recommended/selected badges, and the
user-selection-required note. Frontend rendering applies local redaction before
display.

## Limitations And Future Work

Future phases can add opt-in crawler/indexing, richer live Hugging Face metadata
inspection, dataset download validation, large-dataset profiling, Kaggle support,
legal review workflows for license verification, and a formal human dataset
approval workflow.
