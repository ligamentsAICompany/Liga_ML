import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  IconButton,
  InputLabel,
  Link,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';
import HistoryIcon from '@mui/icons-material/History';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';

import {
  auditBadgeLabel,
  auditEventTitle,
  buildAuditFilters,
  filterAuditEvents,
  formatAuditTimestamp,
  safeAuditLinks,
  severityColor,
  timelineEmptyMessage,
} from '@/lib/audit-timeline';
import type { AuditEvent, AuditFilters, AuditTimelineResponse } from '@/types/audit';
import { apiFetch } from '@/utils/api';

const EMPTY_RESPONSE: AuditTimelineResponse = {
  enabled: true,
  audit_store: null,
  events: [],
};

export default function AuditTimelineButton() {
  const [open, setOpen] = useState(false);
  const [timeline, setTimeline] = useState<AuditTimelineResponse>(EMPTY_RESPONSE);
  const [filters, setFilters] = useState<AuditFilters>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTimeline = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch('/api/audit?limit=200');
      if (!res.ok) throw new Error(`Audit API returned ${res.status}`);
      setTimeline(await res.json() as AuditTimelineResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load audit timeline.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) void loadTimeline();
  }, [open, loadTimeline]);

  useEffect(() => {
    const openTimeline = () => setOpen(true);
    window.addEventListener('liga-open-audit-timeline', openTimeline);
    return () => window.removeEventListener('liga-open-audit-timeline', openTimeline);
  }, []);

  const availableFilters = useMemo(() => buildAuditFilters(timeline.events), [timeline.events]);
  const visibleEvents = useMemo(() => filterAuditEvents(timeline.events, filters), [timeline.events, filters]);

  return (
    <>
      <Button
        size="small"
        variant="outlined"
        startIcon={<HistoryIcon fontSize="small" />}
        onClick={() => setOpen(true)}
        sx={{ textTransform: 'none', borderColor: 'divider', color: 'text.primary' }}
      >
        Timeline
      </Button>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <HistoryIcon fontSize="small" />
          Audit Timeline
          <Box sx={{ flex: 1 }} />
          <IconButton aria-label="Refresh audit timeline" size="small" onClick={loadTimeline} disabled={loading}>
            <RefreshIcon fontSize="small" />
          </IconButton>
          <IconButton aria-label="Close audit timeline" size="small" onClick={() => setOpen(false)}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers sx={{ bgcolor: 'background.default' }}>
          <Stack spacing={2}>
            <Alert severity={timeline.enabled ? 'info' : 'warning'}>
              Internal audit timeline. Events are sanitized before storage and no external observability exporter is configured.
            </Alert>
            {timeline.audit_store?.warning && <Alert severity="warning">{timeline.audit_store.warning}</Alert>}
            {error && <Alert severity="error">{error}</Alert>}

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 1 }}>
              <FilterSelect label="Provider" value={filters.provider} values={availableFilters.providers} onChange={(provider) => setFilters((current) => ({ ...current, provider }))} />
              <FilterSelect label="Category" value={filters.category} values={availableFilters.categories} onChange={(category) => setFilters((current) => ({ ...current, category }))} />
              <FilterSelect label="Severity" value={filters.severity} values={availableFilters.severities} onChange={(severity) => setFilters((current) => ({ ...current, severity }))} />
              <FilterSelect label="Status" value={filters.status} values={availableFilters.statuses} onChange={(status) => setFilters((current) => ({ ...current, status }))} />
            </Box>

            <Divider />

            {!visibleEvents.length ? (
              <Box sx={{ p: 3, border: '1px dashed', borderColor: 'divider', borderRadius: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  {timelineEmptyMessage(filters)}
                </Typography>
              </Box>
            ) : (
              <Stack spacing={1.25}>
                {visibleEvents.map((event) => (
                  <AuditTimelineItem key={event.audit_id} event={event} />
                ))}
              </Stack>
            )}
          </Stack>
        </DialogContent>
      </Dialog>
    </>
  );
}

function FilterSelect({
  label,
  value,
  values,
  onChange,
}: {
  label: string;
  value?: string;
  values: string[];
  onChange: (value: string | undefined) => void;
}) {
  return (
    <FormControl size="small" fullWidth>
      <InputLabel>{label}</InputLabel>
      <Select
        label={label}
        value={value ?? ''}
        onChange={(event) => onChange(event.target.value || undefined)}
      >
        <MenuItem value="">All</MenuItem>
        {values.map((item) => (
          <MenuItem key={item} value={item}>{item}</MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}

function AuditTimelineItem({ event }: { event: AuditEvent }) {
  const links = safeAuditLinks(event);
  const color = severityColor(event.severity);
  return (
    <Box sx={{ p: 1.5, border: '1px solid', borderColor: color === 'error' ? 'error.main' : 'divider', borderRadius: 2, bgcolor: 'background.paper' }}>
      <Stack spacing={0.75}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <Chip size="small" color={color} label={event.category} variant={color === 'default' ? 'outlined' : 'filled'} />
          <Typography variant="body2" sx={{ fontWeight: 800 }}>{auditEventTitle(event)}</Typography>
          <Chip size="small" label={auditBadgeLabel(event)} variant="outlined" />
          <Box sx={{ flex: 1 }} />
          <Typography variant="caption" color="text.secondary">{formatAuditTimestamp(event.timestamp)}</Typography>
        </Box>
        {event.message && <Typography variant="body2">{event.message}</Typography>}
        <Typography variant="caption" color="text.secondary">
          Session {event.session_id}{event.run_id ? ` · Run ${event.run_id}` : ''}{event.tool_name ? ` · Tool ${event.tool_name}` : ''}
        </Typography>
        {(event.dataset_name || event.model_name || event.estimated_cost_usd !== undefined || event.known_cost_usd !== undefined) && (
          <Typography variant="caption" color="text.secondary">
            {event.dataset_name ? `Dataset ${event.dataset_name}` : ''}
            {event.dataset_name && event.model_name ? ' · ' : ''}
            {event.model_name ? `Model ${event.model_name}` : ''}
            {(event.dataset_name || event.model_name) && (event.estimated_cost_usd !== undefined || event.known_cost_usd !== undefined) ? ' · ' : ''}
            {event.estimated_cost_usd !== undefined && event.estimated_cost_usd !== null ? `Estimated $${event.estimated_cost_usd.toFixed(2)}` : ''}
            {event.known_cost_usd !== undefined && event.known_cost_usd !== null ? ` · Known $${event.known_cost_usd.toFixed(2)}` : ''}
          </Typography>
        )}
        {links.length > 0 && (
          <Typography variant="caption">
            {links.map((link, index) => (
              <span key={link.href}>
                {index > 0 ? ' · ' : ''}
                <Link href={link.href} target="_blank" rel="noreferrer">{link.label} link</Link>
              </span>
            ))}
          </Typography>
        )}
        {(event.error_summary || event.severity === 'warning' || event.severity === 'error' || event.severity === 'critical') && (
          <Alert severity={color === 'error' ? 'error' : 'warning'} sx={{ py: 0 }}>
            {event.error_summary || event.status}
          </Alert>
        )}
      </Stack>
    </Box>
  );
}
