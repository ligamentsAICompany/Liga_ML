import type { AuditEvent, AuditFilters } from '../types/audit.js';
import { containsSecretLikeValue, redactJsonLike, redactText } from './redaction.js';

export function chronologicalAuditEvents(events: AuditEvent[]): AuditEvent[] {
  return [...events].sort((a, b) => {
    const aTime = Date.parse(a.timestamp || '') || 0;
    const bTime = Date.parse(b.timestamp || '') || 0;
    return aTime - bTime;
  });
}

export function filterAuditEvents(events: AuditEvent[], filters: AuditFilters): AuditEvent[] {
  return chronologicalAuditEvents(events).map(redactAuditEvent).filter((event) => {
    if (filters.provider && event.provider !== filters.provider) return false;
    if (filters.category && event.category !== filters.category) return false;
    if (filters.severity && event.severity !== filters.severity) return false;
    if (filters.status && event.status !== filters.status) return false;
    return true;
  });
}

export function buildAuditFilters(events: AuditEvent[]): {
  providers: string[];
  categories: string[];
  severities: string[];
  statuses: string[];
} {
  const values = (key: keyof AuditEvent) =>
    Array.from(new Set(events.map((event) => event[key]).filter(Boolean).map(String))).sort();
  return {
    providers: values('provider'),
    categories: values('category'),
    severities: values('severity'),
    statuses: values('status'),
  };
}

export function severityColor(severity: string): 'default' | 'info' | 'warning' | 'error' | 'success' {
  if (severity === 'critical' || severity === 'error') return 'error';
  if (severity === 'warning') return 'warning';
  if (severity === 'info') return 'info';
  return 'default';
}

export function auditEventTitle(event: AuditEvent): string {
  return redactText(event.title?.trim() || event.event_type.replace(/_/g, ' '));
}

export function safeAuditLinks(event: AuditEvent): Array<{ label: string; href: string }> {
  const links: Array<{ label: string; href: string }> = [];
  if (isSafeUrl(event.job_url)) links.push({ label: 'Job', href: event.job_url as string });
  if (isSafeUrl(event.artifact_url)) links.push({ label: 'Artifact', href: event.artifact_url as string });
  return links;
}

export function timelineEmptyMessage(filters: AuditFilters): string {
  return Object.values(filters).some(Boolean)
    ? 'No audit events match the selected filters.'
    : 'No audit events yet. Session, approval, provider job, and result events will appear here.';
}

export function eventContainsSecret(event: AuditEvent): boolean {
  const visible = {
    title: event.title,
    message: event.message,
    error_summary: event.error_summary,
    job_url: event.job_url,
    artifact_url: event.artifact_url,
    safe_metadata: event.safe_metadata,
  };
  return containsSecretLikeValue(visible);
}

export function redactAuditEvent(event: AuditEvent): AuditEvent {
  return redactJsonLike(event);
}

export function formatAuditTimestamp(value: string | null | undefined): string {
  if (!value) return 'Unknown time';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export function auditBadgeLabel(event: AuditEvent): string {
  const parts = [event.provider, event.status].filter(Boolean);
  return parts.join(' · ') || event.category;
}

function isSafeUrl(value: string | null | undefined): boolean {
  if (!value) return false;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' || url.protocol === 'http:' || url.protocol === 's3:' || url.protocol === 'gs:';
  } catch {
    return false;
  }
}
