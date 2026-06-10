import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import {
  SESSION_CAPACITY_ACTION_MESSAGE,
  normalizeSessionCapacityError,
} from '../src/lib/session-capacity.js';

test('capacity detail object produces actionable clear-stale copy', () => {
  const normalized = normalizeSessionCapacityError({
    detail: {
      error: 'session_capacity',
      message: 'old backend message',
      error_type: 'per_user',
      cleanup: { cleared: 0, skipped: 10 },
    },
  });

  assert.equal(normalized.message, SESSION_CAPACITY_ACTION_MESSAGE);
  assert.equal(normalized.canCleanup, true);
});

test('plain capacity strings fall back to actionable copy', () => {
  const normalized = normalizeSessionCapacityError({
    detail: 'Server is at capacity. Please try again later.',
  });

  assert.equal(normalized.message, SESSION_CAPACITY_ACTION_MESSAGE);
  assert.equal(normalized.canCleanup, true);
});

test('new task UIs expose clear stale sessions retry action', () => {
  const welcomeSource = readFileSync(
    join(process.cwd(), 'src/components/WelcomeScreen/WelcomeScreen.tsx'),
    'utf8',
  );
  const sidebarSource = readFileSync(
    join(process.cwd(), 'src/components/SessionSidebar/SessionSidebar.tsx'),
    'utf8',
  );

  assert.match(welcomeSource, /Clear stale sessions/);
  assert.match(welcomeSource, /\/api\/session\/cleanup-stale/);
  assert.match(welcomeSource, /handleStartSession\(\)/);
  assert.match(sidebarSource, /Clear stale sessions/);
  assert.match(sidebarSource, /\/api\/session\/cleanup-stale/);
  assert.match(sidebarSource, /handleNewSession\(\)/);
});

test('layout mounts only the active chat session', () => {
  const layoutSource = readFileSync(
    join(process.cwd(), 'src/components/Layout/AppLayout.tsx'),
    'utf8',
  );

  assert.doesNotMatch(layoutSource, /sessions\.map\(\(s\) =>/);
  assert.match(layoutSource, /activeSessionId/);
  assert.match(layoutSource, /<SessionChat/);
});
