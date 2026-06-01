import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const chatInputSource = readFileSync(new URL('../src/components/Chat/ChatInput.tsx', import.meta.url), 'utf8');

test('chat input accepts Markdown uploads', () => {
  assert.match(chatInputSource, /DATASET_UPLOAD_ACCEPT\s*=\s*['"][^'"]*\.md/);
  assert.match(chatInputSource, /DATASET_UPLOAD_EXTENSIONS[\s\S]*['"]md['"]/);
});

test('uploaded data section renders training metadata', () => {
  assert.match(chatInputSource, /Uploaded Data/);
  assert.match(chatInputSource, /Ready for training/);
  assert.match(chatInputSource, /normalized_row_count/);
  assert.match(chatInputSource, /config_name/);
});

test('uploaded data section covers empty, warning, and malformed metadata states', () => {
  assert.match(chatInputSource, /No uploaded data yet/);
  assert.match(chatInputSource, /Uploaded data is prioritized/);
  assert.match(chatInputSource, /Needs attention/);
  assert.match(chatInputSource, /Unknown file/);
  assert.match(chatInputSource, /Unknown format/);
  assert.match(chatInputSource, /Row count unavailable/);
  assert.doesNotMatch(chatInputSource, /Kaggle-first/);
});
