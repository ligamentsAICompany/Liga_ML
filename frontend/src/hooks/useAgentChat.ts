/**
 * Central hook wiring the Vercel AI SDK's useChat with our SSE-based
 * ChatTransport.
 *
 * In the per-session architecture, each session mounts its own instance
 * of this hook. Side-channel callbacks always update the session's own
 * state via `updateSession()`. If the session is currently active, the
 * store automatically mirrors updates to the flat global fields.
 */
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useChat } from '@ai-sdk/react';
import { type UIMessage, lastAssistantMessageIsCompleteWithApprovalResponses } from 'ai';
import { SSEChatTransport, type SideChannelCallbacks } from '@/lib/sse-chat-transport';
import { registerExplicitToolApprovals } from '@/lib/explicit-tool-approvals';
import { loadMessages, saveMessages } from '@/lib/chat-message-store';
import { saveBackendMessages } from '@/lib/backend-message-store';
import { saveResearch, loadResearch, clearResearch, RESEARCH_MAX_STEPS } from '@/lib/research-store';
import { llmMessagesToUIMessages } from '@/lib/convert-llm-messages';
import { apiFetch } from '@/utils/api';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import { useLayoutStore } from '@/store/layoutStore';
import { logger } from '@/utils/logger';
import { appendTrainingResultSummary, buildVertexStateMarkdown, createVertexRunPanel } from '@/lib/vertex-job-panel';
import { createTrainingPlannerPanel } from '@/lib/training-planner-panel';
import { createDatasetDiscoveryPanel } from '@/lib/dataset-discovery-panel';
import { shouldMarkModelUnavailable, normalizeLlmErrorType } from '@/lib/llm-error-recovery';
import type { ToolStateChangeEventData } from '@/types/events';

interface UseAgentChatOptions {
  sessionId: string;
  isActive: boolean;
  onReady?: () => void;
  onError?: (error: string) => void;
  onSessionDead?: (sessionId: string) => void;
}

const STREAMABLE_TOOLS = new Set(['hf_jobs', 'gcp_vertex_jobs', 'sandbox', 'bash']);

function appendPanelOutput(existing: string, next: string): string {
  if (!existing) return next;
  if (!next) return existing;
  return `${existing}\n${next}`;
}

function createPreflightPanelData(toolName: string, output: unknown, args?: Record<string, unknown>) {
  if (toolName === 'training_planner') {
    const panel = createTrainingPlannerPanel(output ?? args ?? {});
    return {
      title: 'Training Planner',
      output: { content: panel.markdown, language: 'markdown' },
      ...(args ? { input: { content: JSON.stringify(args, null, 2), language: 'json' } } : {}),
    };
  }
  if (toolName === 'dataset_discovery') {
    const panel = createDatasetDiscoveryPanel(output ?? args ?? {});
    return {
      title: 'Dataset Discovery',
      output: { content: panel.markdown, language: 'markdown' },
      ...(args ? { input: { content: JSON.stringify(args, null, 2), language: 'json' } } : {}),
    };
  }
  return null;
}

export function useAgentChat({ sessionId, isActive, onReady, onError, onSessionDead }: UseAgentChatOptions) {
  const callbacksRef = useRef({ onReady, onError, onSessionDead });
  callbacksRef.current = { onReady, onError, onSessionDead };

  const isActiveRef = useRef(isActive);
  isActiveRef.current = isActive;

  const { setNeedsAttention, updateSessionYolo, markSessionModelUnavailable } = useSessionStore();

  // Helper: update this session's state (mirrors to globals if active)
  const updateSession = useAgentStore.getState().updateSession;

  // -- Build side-channel callbacks (stable ref) --------------------------
  const sideChannel = useMemo<SideChannelCallbacks>(
    () => ({
      onReady: () => {
        updateSession(sessionId, { isProcessing: false });
        if (isActiveRef.current) {
          useAgentStore.getState().setConnected(true);
        }
        useSessionStore.getState().setSessionActive(sessionId, true);
        callbacksRef.current.onReady?.();
      },
      onShutdown: () => {
        updateSession(sessionId, { isProcessing: false });
        if (isActiveRef.current) {
          useAgentStore.getState().setConnected(false);
        }
      },
      onRequestStart: (requestId: string) => {
        useAgentStore.getState().clearLlmErrorForNewRequest(sessionId, requestId);
      },
      onRequestSuccess: (requestId: string) => {
        useAgentStore.getState().clearLlmErrorForSuccessfulRequest(sessionId, requestId);
      },
      onError: (error: string, data?: Record<string, unknown>) => {
        updateSession(sessionId, { isProcessing: false });
        const errorType = typeof data?.error_type === 'string' ? data.error_type : null;
        const model = typeof data?.model === 'string' ? data.model : null;
        if (errorType || model) {
          const record = useAgentStore.getState().reportLlmHealthError({
            error,
            errorType,
            model,
            provider: typeof data?.provider === 'string' ? data.provider : null,
            sessionId: typeof data?.session_id === 'string' ? data.session_id : sessionId,
            requestId: typeof data?.request_id === 'string' ? data.request_id : null,
            turnId: typeof data?.turn_id === 'string' || typeof data?.turn_id === 'number' ? data.turn_id : null,
            timestamp: typeof data?.timestamp === 'string' ? data.timestamp : undefined,
          });
          if (record.model && shouldMarkModelUnavailable(normalizeLlmErrorType(record.errorType))) {
            markSessionModelUnavailable(sessionId, record.model, {
              model: record.model,
              errorType: record.errorType,
              message: record.message,
              timestamp: record.timestamp,
            });
          }
        }
        callbacksRef.current.onError?.(error);
      },
      onProcessing: () => {
        updateSession(sessionId, {
          isProcessing: true,
          activityStatus: { type: 'thinking' },
        });
      },
      onProcessingDone: () => {
        updateSession(sessionId, { isProcessing: false });
      },
      onUndoComplete: () => {
        updateSession(sessionId, { isProcessing: false });
      },
      onCompacted: (oldTokens: number, newTokens: number) => {
        logger.log(`Context compacted: ${oldTokens} -> ${newTokens} tokens`);
      },
      onPlanUpdate: (plan) => {
        const typed = plan as Array<{ id: string; content: string; status: 'pending' | 'in_progress' | 'completed' }>;
        updateSession(sessionId, { plan: typed });
        if (isActiveRef.current && !useLayoutStore.getState().isRightPanelOpen) {
          useLayoutStore.getState().setRightPanelOpen(true);
        }
      },
      onToolLog: (tool: string, log: string, agentId?: string, label?: string) => {
        // Research sub-agent: parse stats vs step logs (per-agent)
        if (tool === 'research') {
          const aid = agentId || 'research';
          const sessState = useAgentStore.getState().getSessionState(sessionId);
          const agents = { ...sessState.researchAgents };
          const agent = agents[aid] || { label: label || 'research', steps: [], stats: { toolCount: 0, tokenCount: 0, startedAt: null, finalElapsed: null } };

          if (log === 'Starting research sub-agent...') {
            agents[aid] = {
              label: label || 'research',
              steps: [],
              stats: { toolCount: 0, tokenCount: 0, startedAt: Date.now(), finalElapsed: null },
            };
            // Also update legacy flat fields (aggregate of all agents)
            const allSteps = Object.values(agents).flatMap(a => a.steps);
            const anyRunning = Object.values(agents).some(a => a.stats.startedAt !== null);
            updateSession(sessionId, {
              researchAgents: agents,
              researchSteps: allSteps.slice(-RESEARCH_MAX_STEPS),
              researchStats: anyRunning ? agents[aid].stats : sessState.researchStats,
              activityStatus: { type: 'tool', toolName: 'research', description: label || log },
            });
            saveResearch(sessionId, allSteps.slice(-RESEARCH_MAX_STEPS), agents[aid].stats);
          } else if (log.startsWith('tokens:')) {
            agent.stats = { ...agent.stats, tokenCount: parseInt(log.slice(7), 10) };
            agents[aid] = agent;
            updateSession(sessionId, { researchAgents: agents });
          } else if (log.startsWith('tools:')) {
            agent.stats = { ...agent.stats, toolCount: parseInt(log.slice(6), 10) };
            agents[aid] = agent;
            updateSession(sessionId, { researchAgents: agents });
          } else if (log === 'Research complete.') {
            const elapsed = agent.stats.startedAt
              ? Math.round((Date.now() - agent.stats.startedAt) / 1000)
              : null;
            agent.stats = { ...agent.stats, startedAt: null, finalElapsed: elapsed };
            agents[aid] = agent;
            const anyRunning = Object.values(agents).some(a => a.stats.startedAt !== null);
            updateSession(sessionId, {
              researchAgents: agents,
              researchStats: anyRunning ? sessState.researchStats : agent.stats,
              activityStatus: { type: 'tool', toolName: 'research', description: log },
            });
            // Clear persistence only when ALL agents are done
            if (!anyRunning) clearResearch(sessionId);
          } else {
            // Regular tool call step — append to this agent
            agent.steps = [...agent.steps, log].slice(-RESEARCH_MAX_STEPS);
            agents[aid] = agent;
            const allSteps = Object.values(agents).flatMap(a => a.steps);
            updateSession(sessionId, {
              researchAgents: agents,
              researchSteps: allSteps.slice(-RESEARCH_MAX_STEPS),
              activityStatus: { type: 'tool', toolName: 'research', description: log },
            });
            saveResearch(sessionId, allSteps.slice(-RESEARCH_MAX_STEPS), agent.stats);
          }
          return;
        }

        if (!STREAMABLE_TOOLS.has(tool)) return;

        const sessState = useAgentStore.getState().getSessionState(sessionId);
        const existingOutput = sessState.panelData?.output?.content || '';

        const rawNewContent = appendPanelOutput(existingOutput, log);
        const newContent = tool === 'gcp_vertex_jobs'
          ? appendTrainingResultSummary(rawNewContent)
          : rawNewContent;

        if (!sessState.panelData) {
          const title = tool === 'bash' || tool === 'sandbox'
            ? 'Sandbox'
            : tool === 'gcp_vertex_jobs'
              ? 'Vertex AI Job Output'
              : 'Job Output';
          const language = tool === 'gcp_vertex_jobs' ? 'markdown' : 'text';
          updateSession(sessionId, {
            panelData: { title, output: { content: newContent, language } },
            panelView: 'output',
          });
        } else {
          const language = tool === 'gcp_vertex_jobs'
            ? 'markdown'
            : sessState.panelData.output?.language || 'text';
          updateSession(sessionId, {
            panelData: { ...sessState.panelData, output: { content: newContent, language } },
            panelView: 'output',
          });
        }

        if (isActiveRef.current && !useLayoutStore.getState().isRightPanelOpen) {
          useLayoutStore.getState().setRightPanelOpen(true);
        }
      },
      onConnectionChange: (connected: boolean) => {
        if (isActiveRef.current) useAgentStore.getState().setConnected(connected);
      },
      onSessionDead: (deadSessionId: string) => {
        logger.warn(`Session ${deadSessionId} dead, removing`);
        callbacksRef.current.onSessionDead?.(deadSessionId);
      },
      onApprovalRequired: (tools) => {
        if (!tools.length) return;
        setNeedsAttention(sessionId, true);

        const store = useAgentStore.getState();
        for (const tool of tools) {
          store.setToolBudgetBlock(
            tool.tool_call_id,
            tool.auto_approval_blocked
              ? {
                  reason: tool.block_reason ?? null,
                  estimatedCostUsd: tool.estimated_cost_usd ?? null,
                  remainingCapUsd: tool.remaining_cap_usd ?? null,
                }
              : null,
          );
        }

        updateSession(sessionId, { activityStatus: { type: 'waiting-approval' } });

        // Build panel data for this session's pending approval
        const firstTool = tools[0];
        const args = firstTool.arguments as Record<string, string | undefined>;

        let panelUpdate: Partial<import('@/store/agentStore').PerSessionState> | undefined;
        if (firstTool.tool === 'hf_jobs' && args.script) {
          panelUpdate = {
            panelData: {
              title: 'Script',
              script: { content: args.script, language: 'python' },
              parameters: firstTool.arguments as Record<string, unknown>,
            },
            panelView: 'script' as const,
            panelEditable: true,
          };
        } else if (firstTool.tool === 'gcp_vertex_jobs') {
          const vertexPanel = createVertexRunPanel(firstTool.arguments as Record<string, unknown>);
          if (vertexPanel) {
            panelUpdate = {
              panelData: vertexPanel.data,
              panelView: vertexPanel.view,
              panelEditable: vertexPanel.editable,
            };
          }
        } else if (firstTool.tool === 'hf_repo_files' && args.content) {
          const filename = args.path || 'file';
          panelUpdate = {
            panelData: {
              title: filename.split('/').pop() || 'Content',
              script: { content: args.content, language: filename.endsWith('.py') ? 'python' : 'text' },
              parameters: firstTool.arguments as Record<string, unknown>,
            },
          };
        } else if (firstTool.tool === 'training_planner' || firstTool.tool === 'dataset_discovery') {
          panelUpdate = {
            panelData: createPreflightPanelData(firstTool.tool, null, firstTool.arguments as Record<string, unknown>),
            panelView: 'output' as const,
          };
        } else {
          panelUpdate = {
            panelData: {
              title: firstTool.tool,
              output: { content: JSON.stringify(firstTool.arguments, null, 2), language: 'json' },
            },
            panelView: 'output' as const,
          };
        }
        if (panelUpdate) updateSession(sessionId, panelUpdate);

        if (isActiveRef.current) {
          useLayoutStore.getState().setRightPanelOpen(true);
          useLayoutStore.getState().setLeftSidebarOpen(false);
        }
      },
      onToolCallPanel: (toolName: string, args: Record<string, unknown>) => {
        if (toolName === 'hf_jobs' && args.operation && args.script) {
          updateSession(sessionId, {
            panelData: {
              title: 'Script',
              script: { content: String(args.script), language: 'python' },
              parameters: args,
            },
            panelView: 'script',
          });
          if (isActiveRef.current) {
            useLayoutStore.getState().setRightPanelOpen(true);
            useLayoutStore.getState().setLeftSidebarOpen(false);
          }
        } else if (toolName === 'gcp_vertex_jobs') {
          const vertexPanel = createVertexRunPanel(args);
          if (!vertexPanel) return;
          updateSession(sessionId, {
            panelData: vertexPanel.data,
            panelView: vertexPanel.view,
            panelEditable: vertexPanel.editable,
          });
          if (isActiveRef.current) {
            useLayoutStore.getState().setRightPanelOpen(true);
            useLayoutStore.getState().setLeftSidebarOpen(false);
          }
        } else if (toolName === 'hf_repo_files' && args.operation === 'upload' && args.content) {
          updateSession(sessionId, {
            panelData: {
              title: `File Upload: ${String(args.path || 'unnamed')}`,
              script: { content: String(args.content), language: String(args.path || '').endsWith('.py') ? 'python' : 'text' },
              parameters: args,
            },
          });
          if (isActiveRef.current) {
            useLayoutStore.getState().setRightPanelOpen(true);
            useLayoutStore.getState().setLeftSidebarOpen(false);
          }
        } else if (toolName === 'training_planner' || toolName === 'dataset_discovery') {
          updateSession(sessionId, {
            panelData: createPreflightPanelData(toolName, null, args),
            panelView: 'output',
          });
          if (isActiveRef.current) {
            useLayoutStore.getState().setRightPanelOpen(true);
            useLayoutStore.getState().setLeftSidebarOpen(false);
          }
        } else if (toolName === 'bash' && args.command) {
          updateSession(sessionId, {
            panelData: {
              title: 'Sandbox',
              script: { content: String(args.command), language: 'bash' },
            },
            panelView: 'output',
          });
        }
      },
      onToolOutputPanel: (toolName: string, _toolCallId: string, output: string, success: boolean) => {
        const sessState = useAgentStore.getState().getSessionState(sessionId);
        if (toolName === 'hf_jobs' && output) {
          updateSession(sessionId, {
            panelData: sessState.panelData
              ? { ...sessState.panelData, output: { content: output, language: 'markdown' } }
              : { title: 'Output', output: { content: output, language: 'markdown' } },
            panelView: !success ? 'output' : sessState.panelView,
          });
        } else if (toolName === 'gcp_vertex_jobs' && output) {
          const content = appendTrainingResultSummary(output);
          updateSession(sessionId, {
            panelData: sessState.panelData
              ? { ...sessState.panelData, output: { content, language: 'markdown' } }
              : { title: 'Vertex AI Job Output', output: { content, language: 'markdown' } },
            panelView: !success ? 'output' : sessState.panelView,
            panelEditable: false,
          });
          if (isActiveRef.current && !useLayoutStore.getState().isRightPanelOpen) {
            useLayoutStore.getState().setRightPanelOpen(true);
          }
        } else if ((toolName === 'training_planner' || toolName === 'dataset_discovery') && output) {
          updateSession(sessionId, {
            panelData: createPreflightPanelData(toolName, output),
            panelView: 'output',
            panelEditable: false,
          });
          if (isActiveRef.current && !useLayoutStore.getState().isRightPanelOpen) {
            useLayoutStore.getState().setRightPanelOpen(true);
          }
        } else if (toolName === 'bash') {
          if (!success) {
            updateSession(sessionId, { panelView: 'output' });
          }
        }
      },
      onToolStateChange: (state: ToolStateChangeEventData) => {
        if (state.tool !== 'gcp_vertex_jobs' || !state.tool_call_id) return;

        const store = useAgentStore.getState();
        store.setJobRuntimeState(state.tool_call_id, {
          tool: state.tool,
          state: state.state,
          jobName: state.jobName,
          jobUrl: state.jobUrl,
          outputDir: state.outputDir,
        });
        if (state.jobUrl) {
          store.setJobUrl(state.tool_call_id, state.jobUrl);
        }

        const sessState = store.getSessionState(sessionId);
        const stateMarkdown = buildVertexStateMarkdown({
          state: state.state,
          jobName: state.jobName,
          jobUrl: state.jobUrl,
          outputDir: state.outputDir,
        });
        if (!stateMarkdown) return;

        const existingOutput = sessState.panelData?.output?.content || '';
        const rawContent = existingOutput.includes('## Vertex AI Job State')
          ? existingOutput.replace(/## Vertex AI Job State[\s\S]*$/m, stateMarkdown)
          : appendPanelOutput(existingOutput, stateMarkdown);
        const content = appendTrainingResultSummary(rawContent);

        updateSession(sessionId, {
          panelData: sessState.panelData
            ? { ...sessState.panelData, output: { content, language: 'markdown' } }
            : { title: 'Vertex AI Job Output', output: { content, language: 'markdown' } },
          panelView: sessState.panelData?.script ? sessState.panelView : 'output',
          panelEditable: false,
        });

        if (isActiveRef.current && !useLayoutStore.getState().isRightPanelOpen) {
          useLayoutStore.getState().setRightPanelOpen(true);
        }
      },
      onStreaming: () => {
        updateSession(sessionId, { activityStatus: { type: 'streaming' } });
      },
      onToolRunning: (toolName: string, description?: string) => {
        const updates: Partial<import('@/store/agentStore').PerSessionState> = {
          activityStatus: { type: 'tool', toolName, description },
        };
        // Clear research steps + stats when a new research call starts
        if (toolName === 'research') {
          updates.researchSteps = [];
          updates.researchStats = { toolCount: 0, tokenCount: 0, startedAt: null, finalElapsed: null };
        }
        updateSession(sessionId, updates);
      },
      onInterrupted: () => { /* no-op — handled by stop() caller */ },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId],
  );

  // -- Create transport (one per session, stable for lifetime) ------------
  const transportRef = useRef<SSEChatTransport | null>(null);
  if (!transportRef.current) {
    transportRef.current = new SSEChatTransport(sessionId, sideChannel);
  }

  // Keep side-channel callbacks in sync
  useEffect(() => {
    transportRef.current?.updateSideChannel(sideChannel);
  }, [sideChannel]);

  // Destroy transport on unmount
  useEffect(() => {
    return () => {
      transportRef.current?.destroy();
      transportRef.current = null;
    };
  }, []);

  // -- Restore persisted messages for this session ------------------------
  const initialMessages = useMemo(
    () => loadMessages(sessionId),
    [sessionId],
  );

  // -- Ref for chat actions (used by sideChannel callbacks) ---------------
  const chatActionsRef = useRef<{
    setMessages: ((msgs: UIMessage[]) => void) | null;
    messages: UIMessage[];
  }>({ setMessages: null, messages: [] });

  // -- useChat from Vercel AI SDK -----------------------------------------
  const chat = useChat({
    id: sessionId,
    messages: initialMessages,
    transport: transportRef.current!,
    experimental_throttle: 80,
    // On mount, the SDK calls transport.reconnectToStream() which checks
    // is_processing and subscribes to the live event stream if the agent
    // is mid-turn.  Without this, page refresh kills live updates.
    resume: true,
    // After all approval responses are set, auto-send to continue the agent loop.
    // Without this, addToolApprovalResponse only updates the UI — it won't trigger
    // sendMessages on the transport.
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
    onError: (error) => {
      updateSession(sessionId, { isProcessing: false });
      // Premium-model daily cap: open the cap dialog instead of the generic error
      // banner. Transport marks the error with this sentinel.
      if (error.message === 'CLAUDE_QUOTA_EXHAUSTED') {
        if (isActiveRef.current) {
          useAgentStore.getState().setClaudeQuotaExhausted(true);
        }
        return;
      }
      logger.error('useChat error:', error);
    },
  });

  // Keep chatActionsRef in sync every render
  chatActionsRef.current.setMessages = chat.setMessages;
  chatActionsRef.current.messages = chat.messages;

  // -- Hydrate from backend on mount (page refresh recovery) --------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [msgsRes, infoRes] = await Promise.all([
          apiFetch(`/api/session/${sessionId}/messages`),
          apiFetch(`/api/session/${sessionId}`),
        ]);
        if (cancelled) return;

        // If both endpoints say "not found", the backend lost this session
        // (typically: Space restarted). Fire onSessionDead so AppLayout
        // can flag it for the catch-up banner.
        if (infoRes.status === 404 && msgsRes.status === 404) {
          callbacksRef.current.onSessionDead?.(sessionId);
          return;
        }

        let pendingIds: Set<string> | undefined;
        let backendIsProcessing = false;
        if (infoRes.ok) {
          const info = await infoRes.json();
          backendIsProcessing = !!info.is_processing;
          if (info.pending_approval && Array.isArray(info.pending_approval)) {
            pendingIds = new Set(
              info.pending_approval.map((t: { tool_call_id: string }) => t.tool_call_id)
            );
            if (pendingIds.size > 0) {
              setNeedsAttention(sessionId, true);
            }
          }
        }

        if (msgsRes.ok) {
          const data = await msgsRes.json();
          if (cancelled || !Array.isArray(data) || data.length === 0) return;
          // Cache the raw backend messages so we can restore this session
          // into a fresh backend if the Space restarts.
          saveBackendMessages(sessionId, data);
          const uiMsgs = llmMessagesToUIMessages(data, pendingIds, chatActionsRef.current.messages);
          if (uiMsgs.length > 0) {
            chat.setMessages(uiMsgs);
            saveMessages(sessionId, uiMsgs);
          }
        }

        // Use the backend's is_processing flag as the source of truth.
        // Message-based inference doesn't work because completed tool
        // results make tools look "done" even when the agent is still
        // mid-turn and about to call more tools.
        if (backendIsProcessing) {
          // Restore research sub-agent state alongside isProcessing in one
          // atomic update so the UI never sees isProcessing=false with stale
          // tool states (which would coerce them to 'output-available').
          const savedResearch = loadResearch(sessionId);
          updateSession(sessionId, {
            isProcessing: true,
            activityStatus: savedResearch?.stats.startedAt
              ? { type: 'tool', toolName: 'research', description: 'Resuming research...' }
              : { type: 'thinking' },
            ...(savedResearch && {
              researchSteps: savedResearch.steps,
              researchStats: savedResearch.stats,
            }),
          });
        } else if (pendingIds && pendingIds.size > 0) {
          updateSession(sessionId, { activityStatus: { type: 'waiting-approval' } });
          clearResearch(sessionId);
        } else {
          clearResearch(sessionId);
        }
      } catch {
        /* backend unreachable -- localStorage fallback is fine */
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Re-hydrate + reconnect on wake from sleep ----------------------------
  // The Vercel AI SDK only calls reconnectToStream() on mount, NOT on
  // visibility change.  So when the browser wakes from sleep and the SSE
  // stream is dead, we must manually:
  //   1. Re-hydrate messages (one-shot fetch from backend)
  //   2. Subscribe to live events via GET /api/events/{sessionId}
  //   3. Pipe those events through the side-channel callbacks for real-time UI
  //   4. Poll messages every few seconds so chat.setMessages stays in sync
  const reconnectAbortRef = useRef<AbortController | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    /** Fetch latest messages from backend and push into the SDK. */
    const hydrateMessages = async () => {
      try {
        const [msgsRes, infoRes] = await Promise.all([
          apiFetch(`/api/session/${sessionId}/messages`),
          apiFetch(`/api/session/${sessionId}`),
        ]);
        if (!msgsRes.ok) return null;
        const data = await msgsRes.json();
        if (!Array.isArray(data) || data.length === 0) return null;

        // Cache the raw backend messages so we can restore this session
        // into a fresh backend if the Space restarts.
        saveBackendMessages(sessionId, data);

        let pendingIds: Set<string> | undefined;
        if (infoRes.ok) {
          const info = await infoRes.json();
          if (info.pending_approval && Array.isArray(info.pending_approval)) {
            pendingIds = new Set(
              info.pending_approval.map((t: { tool_call_id: string }) => t.tool_call_id)
            );
            if (pendingIds.size > 0) setNeedsAttention(sessionId, true);
          }
          if (info.auto_approval) {
            updateSessionYolo(sessionId, info.auto_approval);
          }
          return { data, pendingIds, info };
        }
        return { data, pendingIds, info: null };
      } catch {
        return null;
      }
    };

    /** Stop any running reconnection (event stream + poll). */
    const stopReconnect = () => {
      reconnectAbortRef.current?.abort();
      reconnectAbortRef.current = null;
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };

    /** Read the event stream from GET /api/events and forward to side-channel. */
    const consumeEventStream = async (signal: AbortSignal) => {
      try {
        const lastEventKey = `hf-agent-last-event:${sessionId}`;
        const lastSeq = localStorage.getItem(lastEventKey);
        const qs = lastSeq ? `?after=${encodeURIComponent(lastSeq)}` : '';
        const res = await apiFetch(`/api/events/${sessionId}${qs}`, {
          headers: { 'Accept': 'text/event-stream' },
          signal,
        });
        if (!res.ok || !res.body) return;

        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        let buf = '';
        let eventId: string | null = null;
        let eventData = '';
        const dispatch = async () => {
          if (!eventData.trim()) {
            eventId = null;
            eventData = '';
            return false;
          }
          const event = JSON.parse(eventData.trim());
          const seq = event.seq ?? (eventId ? Number(eventId) : undefined);
          if (Number.isFinite(seq)) {
            localStorage.setItem(lastEventKey, String(seq));
          }
          eventId = null;
          eventData = '';
          // Forward to side-channel for real-time UI updates
          const et = event.event_type as string;
          if (et === 'processing') sideChannel.onProcessing();
          else if (et === 'assistant_chunk') sideChannel.onStreaming();
          else if (et === 'tool_call') {
            const t = event.data?.tool as string;
            const d = event.data?.arguments?.description as string | undefined;
            sideChannel.onToolRunning(t, d);
            sideChannel.onToolCallPanel(t, (event.data?.arguments || {}) as Record<string, unknown>);
          } else if (et === 'tool_output') {
            sideChannel.onToolOutputPanel(
              event.data?.tool as string,
              event.data?.tool_call_id as string,
              event.data?.output as string,
              event.data?.success as boolean,
            );
          } else if (et === 'tool_state_change') {
            const state = event.data?.state as string;
            const toolName = event.data?.tool as string;
            const toolCallId = event.data?.tool_call_id as string;
            if (toolCallId) {
              sideChannel.onToolStateChange({
                tool: toolName,
                tool_call_id: toolCallId,
                state,
                jobName: event.data?.jobName as string | undefined,
                jobUrl: event.data?.jobUrl as string | undefined,
                outputDir: event.data?.outputDir as string | undefined,
                trackioSpaceId: event.data?.trackioSpaceId as string | undefined,
                trackioProject: event.data?.trackioProject as string | undefined,
                namespace: event.data?.namespace as string | undefined,
              });
            }
            if (state === 'running' && toolName) sideChannel.onToolRunning(toolName);
          } else if (et === 'turn_complete' || et === 'error' || et === 'interrupted') {
            sideChannel.onProcessingDone();
            stopReconnect();
            // Final hydration to get the complete message state
            const result = await hydrateMessages();
            if (result) {
              const uiMsgs = llmMessagesToUIMessages(result.data, result.pendingIds, chatActionsRef.current.messages);
              if (uiMsgs.length > 0) {
                chat.setMessages(uiMsgs);
                saveMessages(sessionId, uiMsgs);
              }
            }
            return true;
          } else if (et === 'approval_required') {
            sideChannel.onApprovalRequired(
              (event.data?.tools || []) as Array<{
                tool: string;
                arguments: Record<string, unknown>;
                tool_call_id: string;
                auto_approval_blocked?: boolean;
                block_reason?: string | null;
                estimated_cost_usd?: number | null;
                remaining_cap_usd?: number | null;
              }>,
            );
            stopReconnect();
            const result = await hydrateMessages();
            if (result) {
              const uiMsgs = llmMessagesToUIMessages(result.data, result.pendingIds, chatActionsRef.current.messages);
              if (uiMsgs.length > 0) {
                chat.setMessages(uiMsgs);
                saveMessages(sessionId, uiMsgs);
              }
            }
            return true;
          }
          return false;
        };
        while (true) {
          const { value, done } = await reader.read();
          if (done || signal.aborted) break;
          buf += value;
          const lines = buf.split('\n');
          buf = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.replace(/\r$/, '');
            if (trimmed === '') {
              try {
                if (await dispatch()) return;
              } catch { /* ignore parse errors */ }
              continue;
            }
            if (trimmed.startsWith(':')) continue;
            if (trimmed.startsWith('id:')) {
              eventId = trimmed.slice(3).trim();
              continue;
            }
            if (trimmed.startsWith('data:')) {
              eventData += trimmed.slice(5).trimStart() + '\n';
            }
          }
        }
      } catch {
        /* stream ended or aborted */
      }
    };

    const onVisible = async () => {
      if (document.visibilityState !== 'visible') return;

      // Always re-hydrate messages on wake
      const result = await hydrateMessages();
      if (!result) return;

      const { data, pendingIds, info } = result;
      const uiMsgs = llmMessagesToUIMessages(data, pendingIds, chatActionsRef.current.messages);
      if (uiMsgs.length > 0) {
        chat.setMessages(uiMsgs);
        saveMessages(sessionId, uiMsgs);
      }

      // If the backend is still processing, reconnect to the live event stream
      if (info?.is_processing) {
        updateSession(sessionId, { isProcessing: true, activityStatus: { type: 'thinking' } });

        // Stop any previous reconnection
        stopReconnect();

        // Start live event subscription
        const abort = new AbortController();
        reconnectAbortRef.current = abort;
        consumeEventStream(abort.signal);

        // Poll messages every 3 s so the chat message list stays up-to-date
        // (the event stream gives us real-time status but not full message diffs)
        pollTimerRef.current = setInterval(async () => {
          const fresh = await hydrateMessages();
          if (!fresh) return;
          const msgs = llmMessagesToUIMessages(fresh.data, fresh.pendingIds, chatActionsRef.current.messages);

          const currentCount = chatActionsRef.current.messages.length;
          if (msgs.length > currentCount || currentCount === 0) {
            chat.setMessages(msgs);
            saveMessages(sessionId, msgs);
          } 

          // If backend stopped processing, clean up
          if (fresh.info && !fresh.info.is_processing) {
            updateSession(sessionId, { isProcessing: false });
            stopReconnect();
          }
        }, 3000);
      }
    };

    document.addEventListener('visibilitychange', onVisible);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      stopReconnect();
    };
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Persist messages ---------------------------------------------------
  const prevLenRef = useRef(initialMessages.length);
  useEffect(() => {
    if (chat.messages.length === 0) return;
    if (chat.messages.length !== prevLenRef.current) {
      prevLenRef.current = chat.messages.length;
      saveMessages(sessionId, chat.messages);
    } 
  }, [sessionId, chat.messages]);

  // -- Undo last turn (REST call + client-side message removal) -----------
  // With SSE there's no persistent connection to receive the undo_complete
  // event, so we handle message removal on the frontend after a successful
  // REST call to the backend.
  const undoLastTurn = useCallback(async () => {
    try {
      const res = await apiFetch(`/api/undo/${sessionId}`, { method: 'POST' });
      if (!res.ok) {
        logger.error('Undo API returned', res.status);
        return;
      }
      // Remove the last user turn + assistant response from the UI
      const msgs = chatActionsRef.current.messages;
      const setMsgs = chatActionsRef.current.setMessages;
      if (setMsgs && msgs.length > 0) {
        let lastUserIdx = -1;
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].role === 'user') { lastUserIdx = i; break; }
        }
        const updated = lastUserIdx > 0 ? msgs.slice(0, lastUserIdx) : [];
        setMsgs(updated);
        saveMessages(sessionId, updated);
      }
      updateSession(sessionId, { isProcessing: false });
    } catch (e) {
      logger.error('Undo failed:', e);
    }
  }, [sessionId, updateSession]);

  // -- Approve tools ------------------------------------------------------
  const approveTools = useCallback(
    async (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null; edited_script?: string | null; namespace?: string | null }>) => {
      // Store edited scripts so the transport can read them when sendMessages is called
      for (const a of approvals) {
        if (a.edited_script) {
          useAgentStore.getState().setEditedScript(a.tool_call_id, a.edited_script);
        }
      }
      registerExplicitToolApprovals(sessionId, approvals);

      // Update SDK tool state — this triggers sendMessages() via the transport
      for (const a of approvals) {
        chat.addToolApprovalResponse({
          id: `approval-${a.tool_call_id}`,
          approved: a.approved,
          reason: a.approved ? undefined : (a.feedback || 'Rejected by user'),
        });
      }

      setNeedsAttention(sessionId, false);
      const hasApproved = approvals.some(a => a.approved);
      if (hasApproved) {
        updateSession(sessionId, {
          isProcessing: true,
          activityStatus: { type: 'thinking' },
        });
      }

      // Persist updated tool states so a page refresh during execution
      // won't restore stale approval-requested state from localStorage.
      saveMessages(sessionId, chatActionsRef.current.messages);

      return true;
    },
    [sessionId, chat, updateSession, setNeedsAttention],
  );

  // -- Stop (interrupt backend agent loop, keep SSE open for events) --------
  const stop = useCallback(() => {
    // Don't call chat.stop() — keep the SSE stream open so the backend's
    // tool_state_change(cancelled) and interrupted events reach the frontend.
    // The stream closes naturally when the backend sends finish events.
    updateSession(sessionId, { isProcessing: false });
    apiFetch(`/api/interrupt/${sessionId}`, { method: 'POST' }).catch(() => {});
  }, [sessionId, updateSession]);

  // -- Edit message + regenerate from that point ----------------------------
  const editAndRegenerate = useCallback(async (messageId: string, newText: string) => {
    try {
      const msgs = chatActionsRef.current.messages;
      const setMsgs = chatActionsRef.current.setMessages;
      if (!setMsgs) return;

      // Find the target message and compute user message index (0-indexed, skipping system)
      const msgIndex = msgs.findIndex(m => m.id === messageId);
      if (msgIndex < 0) return;

      let userMsgIndex = 0;
      for (let i = 0; i < msgIndex; i++) {
        if (msgs[i].role === 'user') userMsgIndex++;
      }

      // 1. Truncate backend history
      const res = await apiFetch(`/api/truncate/${sessionId}`, {
        method: 'POST',
        body: JSON.stringify({ user_message_index: userMsgIndex }),
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) {
        logger.error('Truncate API returned', res.status);
        return;
      }

      // 2. Truncate frontend messages
      const truncated = msgs.slice(0, msgIndex);
      setMsgs(truncated);
      saveMessages(sessionId, truncated);

      // 3. Send the edited message (reuses existing transport + /api/chat)
      chat.sendMessage({ text: newText, metadata: { createdAt: new Date().toISOString() } });
    } catch (e) {
      logger.error('Edit and regenerate failed:', e);
    }
  }, [sessionId, chat]);

  const refreshMessages = useCallback(async () => {
    try {
      const [msgsRes, infoRes] = await Promise.all([
        apiFetch(`/api/session/${sessionId}/messages`),
        apiFetch(`/api/session/${sessionId}`),
      ]);
      if (!msgsRes.ok) return false;

      const data = await msgsRes.json();
      if (!Array.isArray(data) || data.length === 0) return false;
      saveBackendMessages(sessionId, data);

      let pendingIds: Set<string> | undefined;
      if (infoRes.ok) {
        const info = await infoRes.json();
        if (info.pending_approval && Array.isArray(info.pending_approval)) {
          pendingIds = new Set(
            info.pending_approval.map((t: { tool_call_id: string }) => t.tool_call_id)
          );
          if (pendingIds.size > 0) setNeedsAttention(sessionId, true);
        }
        if (info.auto_approval) {
          updateSessionYolo(sessionId, info.auto_approval);
        }
      }

      const uiMsgs = llmMessagesToUIMessages(
        data,
        pendingIds,
        chatActionsRef.current.messages,
      );
      const setMsgs = chatActionsRef.current.setMessages;
      if (setMsgs && uiMsgs.length > 0) {
        setMsgs(uiMsgs);
        saveMessages(sessionId, uiMsgs);
      }
      return true;
    } catch {
      return false;
    }
  }, [sessionId, setNeedsAttention, updateSessionYolo]);

  return {
    messages: chat.messages,
    sendMessage: chat.sendMessage,
    stop,
    status: chat.status,
    undoLastTurn,
    editAndRegenerate,
    approveTools,
    refreshMessages,
  };
}
