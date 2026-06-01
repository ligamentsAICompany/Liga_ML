export type ExplicitApprovalDecision = {
  approved: boolean;
  feedback?: string | null;
  edited_script?: string | null;
  namespace?: string | null;
};

const explicitApprovalDecisions = new Map<string, ExplicitApprovalDecision>();

function approvalDecisionKey(sessionId: string, toolCallId: string): string {
  return `${sessionId}:${toolCallId}`;
}

export function registerExplicitToolApprovals(
  sessionId: string,
  approvals: Array<{
    tool_call_id: string;
    approved: boolean;
    feedback?: string | null;
    edited_script?: string | null;
    namespace?: string | null;
  }>,
): void {
  for (const approval of approvals) {
    explicitApprovalDecisions.set(approvalDecisionKey(sessionId, approval.tool_call_id), {
      approved: approval.approved,
      feedback: approval.feedback ?? null,
      edited_script: approval.edited_script ?? null,
      namespace: approval.namespace ?? null,
    });
  }
}

export function consumeExplicitApprovalDecision(
  sessionId: string,
  toolCallId: string,
): ExplicitApprovalDecision | null {
  const key = approvalDecisionKey(sessionId, toolCallId);
  const decision = explicitApprovalDecisions.get(key) ?? null;
  if (decision) explicitApprovalDecisions.delete(key);
  return decision;
}

export function clearExplicitToolApprovalsForTesting(): void {
  explicitApprovalDecisions.clear();
}
