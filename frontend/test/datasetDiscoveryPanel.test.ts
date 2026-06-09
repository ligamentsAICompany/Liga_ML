import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { createDatasetDiscoveryPanel } from '../src/lib/dataset-discovery-panel.js';

const toolCallGroupSource = readFileSync('src/components/Chat/ToolCallGroup.tsx', 'utf8');

test('dataset discovery panel renders allowed sources', () => {
  const panel = createDatasetDiscoveryPanel({
    allowedSources: ['huggingface', 'github', 'papers', 'public_web'],
  });

  assert.match(panel.markdown, /Allowed sources/);
  assert.match(panel.markdown, /Hugging Face Datasets/);
  assert.match(panel.markdown, /GitHub/);
  assert.match(panel.markdown, /papers/);
  assert.match(panel.markdown, /public web/);
});

test('dataset discovery panel renders Kaggle only as excluded future work', () => {
  const panel = createDatasetDiscoveryPanel({
    allowedSources: ['huggingface', 'kaggle'],
    excludedSources: ['kaggle'],
  });

  assert.doesNotMatch(panel.allowedSourceLines.join('\n'), /Kaggle/);
  assert.match(panel.excludedSourceLines.join('\n'), /Kaggle \(future work only; not connected\)/);
  assert.match(panel.markdown, /Kaggle \(future work only; not connected\)/);
});

test('dataset discovery panel renders candidate datasets and scores', () => {
  const panel = createDatasetDiscoveryPanel({
    candidates: [
      {
        name: 'Support Tickets',
        source: 'huggingface',
        score: 0.91,
        reason: 'Matches support fine-tuning.',
        risks: ['Verify license.'],
      },
    ],
  });

  assert.match(panel.markdown, /Support Tickets/);
  assert.match(panel.markdown, /score 0\.91/);
  assert.match(panel.markdown, /Verify license\./);
});

test('dataset discovery panel renders enhanced candidate cards and warnings', () => {
  const panel = createDatasetDiscoveryPanel({
    intent: {
      domain: 'hardware_support',
      task_type: 'sft',
      target_provider: 'aws-sagemaker',
      data_modality: 'text',
    },
    selected_candidate: {
      dataset_id: 'public/hardware-support',
    },
    candidates: [
      {
        dataset_id: 'public/hardware-support',
        source: 'huggingface',
        repo_id: 'public/hardware-support',
        title: 'Hardware Support QA',
        license: 'mit',
        license_status: 'clear',
        privacy_status: 'low',
        schema_status: 'compatible',
        row_count: 5000,
        columns: ['instruction', 'output', 'category'],
        text_columns: ['instruction', 'output'],
        quality_score: {
          overall_score: 0.89,
          relevance_score: 0.92,
          license_score: 1,
          safety_score: 0.9,
          schema_score: 1,
        },
        reasons: ['Matches hardware troubleshooting instruction response.'],
        warnings: ['Confirm final dataset choice before launch.'],
        load_dataset_snippet: 'from datasets import load_dataset\nload_dataset("public/hardware-support")',
      },
    ],
  });

  assert.match(panel.markdown, /hardware_support/);
  assert.match(panel.markdown, /Recommended/);
  assert.match(panel.markdown, /Selected/);
  assert.match(panel.markdown, /Overall 0\.89/);
  assert.match(panel.markdown, /License clear/);
  assert.match(panel.markdown, /Privacy low/);
  assert.match(panel.markdown, /Schema compatible/);
  assert.match(panel.markdown, /instruction, output, category/);
  assert.match(panel.markdown, /from datasets import load_dataset/);
  assert.match(panel.markdown, /Confirm final dataset choice/);
});

test('dataset discovery panel renders structured risks and keeps excluded candidates after safe candidates', () => {
  const panel = createDatasetDiscoveryPanel({
    candidates: [
      {
        dataset_id: 'kaggle/ipl',
        source: 'kaggle',
        title: 'Kaggle IPL',
        score: 0.99,
        excluded: true,
        exclusion_reason: 'Kaggle is future work only.',
      },
      {
        dataset_id: 'public/support',
        source: 'huggingface',
        title: 'Public Support',
        score: 0.7,
        risks: [
          {
            category: 'privacy',
            severity: 'warning',
            message: 'Review possible personal data before training.',
          },
        ],
      },
    ],
  });

  assert.match(panel.markdown, /Review possible personal data before training/);
  assert.equal(panel.candidateLines[0].includes('Public Support'), true);
  assert.equal(panel.candidateLines[1].includes('Kaggle IPL'), true);
});

test('dataset discovery panel redacts secrets from structured payloads', () => {
  const panel = createDatasetDiscoveryPanel({
    query: 'hardware support HF_TOKEN=hf_secret',
    candidates: [
      {
        dataset_id: 'safe/support',
        source: 'huggingface',
        title: 'Support sk-test-secret',
        warnings: ['token abc123secret should not show'],
      },
    ],
  });

  assert.doesNotMatch(panel.markdown, /hf_secret|sk-test-secret|abc123secret/);
  assert.match(panel.markdown, /\[REDACTED\]/);
});

test('dataset discovery panel renders user-selection requirement and empty candidates', () => {
  const panel = createDatasetDiscoveryPanel({ candidates: [] });

  assert.match(panel.markdown, /No uploaded dataset is attached/);
  assert.match(panel.markdown, /No candidate datasets supplied yet/);
  assert.match(panel.markdown, /User selection required before training/);
});

test('dataset discovery tool displays a readable label', () => {
  assert.match(toolCallGroupSource, /dataset_discovery/);
  assert.match(toolCallGroupSource, /Dataset Discovery/);
});
