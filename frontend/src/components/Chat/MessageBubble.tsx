import UserMessage from './UserMessage';
import AssistantMessage from './AssistantMessage';
import type { UIMessage } from 'ai';
import { redactUIMessage } from '@/lib/chat-redaction';

interface MessageBubbleProps {
  message: UIMessage;
  isLastTurn?: boolean;
  onUndoTurn?: () => void;
  onEditAndRegenerate?: (messageId: string, newText: string) => void | Promise<void>;
  isProcessing?: boolean;
  isStreaming?: boolean;
  sessionId?: string | null;
  approveTools: (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null }>) => Promise<boolean>;
}

export default function MessageBubble({
  message,
  isLastTurn = false,
  onUndoTurn,
  onEditAndRegenerate,
  isProcessing = false,
  isStreaming = false,
  sessionId,
  approveTools,
}: MessageBubbleProps) {
  const safeMessage = redactUIMessage(message);

  if (message.role === 'user') {
    return (
      <UserMessage
        message={safeMessage}
        isLastTurn={isLastTurn}
        onUndoTurn={onUndoTurn}
        onEditAndRegenerate={onEditAndRegenerate}
        isProcessing={isProcessing}
      />
    );
  }

  if (message.role === 'assistant') {
    return (
      <AssistantMessage
        message={safeMessage}
        isStreaming={isStreaming}
        sessionId={sessionId}
        approveTools={approveTools}
      />
    );
  }

  return null;
}
