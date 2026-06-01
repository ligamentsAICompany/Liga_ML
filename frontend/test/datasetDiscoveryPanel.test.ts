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
