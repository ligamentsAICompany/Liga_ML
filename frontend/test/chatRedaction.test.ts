import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  redactOutboundChatText,
  redactUIMessages,
  sanitizeMessagesMap,
  sanitizeRestoredMessages,
} from '../src/lib/chat-redaction.js';
import { containsSecretLikeValue } from '../src/lib/redaction.js';

const HF_TOKEN = 'hf_FAKE_TEST_TOKEN_1234567890';
const AWS_SECRET = 'FAKEAWSSECRET1234567890';
const MONGO_URI = 'mongodb+srv://fake_user:fake_password@example.mongodb.net/test';
const PRIVATE_KEY = '-----BEGIN PRIVATE KEY-----FAKE-----END PRIVATE KEY-----';
const RAW_MARKERS = [
  HF_TOKEN,
  AWS_SECRET,
  'fake_password',
  'BEGIN PRIVATE KEY',
];

function assertNoRawMarkers(value: unknown): void {
  const serialized = JSON.stringify(value);
  for (const marker of RAW_MARKERS) {
    assert.equal(serialized.includes(marker), false, `unexpected raw marker: ${marker}`);
  }
}

test('redacts user chat messages before display or storage', () => {
  const messages = redactUIMessages([
    {
      id: 'user-1',
      role: 'user',
      parts: [{
        type: 'text',
        text: [
          `HF_TOKEN=${HF_TOKEN}`,
          `AWS_SECRET_ACCESS_KEY=${AWS_SECRET}`,
          `MONGODB_URI=${MONGO_URI}`,
          `PRIVATE_KEY=${PRIVATE_KEY}`,
          'Keep normal instruction text.',
        ].join('\n'),
      }],
    },
  ]);

  assertNoRawMarkers(messages);
  assert.match(JSON.stringify(messages), /\[REDACTED\]/);
  assert.match(JSON.stringify(messages), /Keep normal instruction text/);
});

test('redacts assistant and tool message parts while preserving normal text', () => {
  const messages = redactUIMessages([
    {
      id: 'assistant-1',
      role: 'assistant',
      parts: [
        { type: 'text', text: `I will not reveal Authorization: Bearer ${HF_TOKEN}` },
        {
          type: 'dynamic-tool',
          toolCallId: 'tool-1',
          toolName: 'training_planner',
          state: 'output-available',
          input: { prompt: `token=${HF_TOKEN}`, normal: 'dataset summary' },
          output: `AWS_SECRET_ACCESS_KEY=${AWS_SECRET}`,
        },
      ],
    },
  ]);

  assertNoRawMarkers(messages);
  assert.match(JSON.stringify(messages), /dataset summary/);
  assert.equal(containsSecretLikeValue(messages), false);
});

test('sanitizes hf-agent-messages maps before localStorage persistence', () => {
  const sanitized = sanitizeMessagesMap({
    'session-1': [
      {
        id: 'user-1',
        role: 'user',
        parts: [{ type: 'text', text: `MONGODB_URI=${MONGO_URI}` }],
      },
    ],
  });

  assertNoRawMarkers(sanitized);
  assert.match(JSON.stringify(sanitized), /\[REDACTED\]/);
});

test('migrates existing raw localStorage message entries on read', () => {
  const restored = sanitizeRestoredMessages([
    {
      id: 'legacy-user',
      role: 'user',
      parts: [{ type: 'text', text: `PRIVATE_KEY=${PRIVATE_KEY}` }],
    },
  ]);

  assertNoRawMarkers(restored);
  assert.match(JSON.stringify(restored), /\[REDACTED\]/);
});

test('session restore uses redacted content and keeps safe prose', () => {
  const restored = sanitizeRestoredMessages([
    {
      id: 'session-user',
      role: 'user',
      parts: [{ type: 'text', text: `Please plan safely with HF_TOKEN=${HF_TOKEN}. Normal text survives.` }],
    },
  ]);

  assertNoRawMarkers(restored);
  assert.match(JSON.stringify(restored), /Normal text survives/);
});

test('outbound chat text is redacted before normal backend submission', () => {
  const outbound = redactOutboundChatText(
    `Plan safely with HF_TOKEN=${HF_TOKEN} and MONGODB_URI=${MONGO_URI}.`,
  );

  assertNoRawMarkers(outbound);
  assert.match(outbound, /Plan safely/);
});
