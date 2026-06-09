import type { UIMessage } from 'ai';
import { redactJsonLike, redactText } from './redaction.js';

export type MessagesMap = Record<string, UIMessage[]>;

export function redactUIMessage(message: UIMessage): UIMessage {
  return redactJsonLike(message);
}

export function redactUIMessages(messages: UIMessage[]): UIMessage[] {
  return messages.map((message) => redactUIMessage(message));
}

export function sanitizeMessagesMap(map: MessagesMap): MessagesMap {
  return Object.fromEntries(
    Object.entries(map).map(([sessionId, messages]) => [
      sessionId,
      Array.isArray(messages) ? redactUIMessages(messages) : [],
    ]),
  );
}

export function sanitizeRestoredMessages(messages: UIMessage[]): UIMessage[] {
  return redactUIMessages(messages);
}

export function redactOutboundChatText(text: string): string {
  return redactText(text);
}
