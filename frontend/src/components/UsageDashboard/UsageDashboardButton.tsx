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
  IconButton,
  Link,
  Stack,
  Typography,
} from '@mui/material';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';

import { buildProviderCards, formatUsd, providerLabel, usageEntryTitle } from '@/lib/usage-dashboard';
import { redactJsonLike, redactText } from '@/lib/redaction';
import type { UsageSummary } from '@/types/usage';
import { apiFetch } from '@/utils/api';

const EMPTY_SUMMARY: UsageSummary = {
  total_estimated_cost_usd: 0,
  total_known_cost_usd: 0,
  cost_by_provider: {},
  cost_by_session: {},
  cost_by_run: {},
  recent_usage_entries: [],
  quota_warnings: [],
  budget_warnings: [],
  provider_readiness: {},
  usage_store: null,
};

export default function UsageDashboardButton() {
  const [open, setOpen] = useState(false);
  const [summary, setSummary] = useState<UsageSummary>(EMPTY_SUMMARY);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch('/api/usage/summary?limit=100');
      if (!res.ok) throw new Error(`Usage API returned ${res.status}`);
      setSummary(redactJsonLike(await res.json() as UsageSummary));
    } catch (e) {
      setError(redactText(e instanceof Error ? e.message : 'Failed to load usage dashboard.'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) void loadSummary();
  }, [open, loadSummary]);

  const providerCards = useMemo(() => buildProviderCards(summary), [summary]);
  const hasEntries = summary.recent_usage_entries.length > 0;

  return (
    <>
      <Button
        size="small"
        variant="outlined"
        startIcon={<AccountBalanceWalletIcon fontSize="small" />}
        onClick={() => setOpen(true)}
        sx={{ textTransform: 'none', borderColor: 'divider', color: 'text.primary' }}
      >
        Usage
      </Button>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AccountBalanceWalletIcon fontSize="small" />
          Usage, Billing, Quota, and Budget
          <Box sx={{ flex: 1 }} />
          <IconButton aria-label="Refresh usage dashboard" size="small" onClick={loadSummary} disabled={loading}>
            <RefreshIcon fontSize="small" />
          </IconButton>
          <IconButton aria-label="Close usage dashboard" size="small" onClick={() => setOpen(false)}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers sx={{ bgcolor: 'background.default' }}>
          <Stack spacing={2}>
            <Alert severity="info">
              Estimated cost, not final bill. Actual provider billing may differ. No live billing API configured.
              Quota status may be unknown unless provider reports it.
            </Alert>

            {summary.usage_store?.warning && <Alert severity="warning">{summary.usage_store.warning}</Alert>}
            {error && <Alert severity="error">{error}</Alert>}

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 1.5 }}>
              <MetricCard label="Total estimated usage" value={formatUsd(summary.total_estimated_cost_usd)} helper="Estimated cost, not final bill" />
              <MetricCard label="Known actual usage" value={formatUsd(summary.total_known_cost_usd)} helper="Unknown when providers do not report billing" />
              <MetricCard label="Usage store" value={summary.usage_store?.durable ? 'Durable MongoDB' : 'In-memory fallback'} helper={summary.usage_store?.enabled ? 'Dashboard enabled' : 'Dashboard disabled'} />
            </Box>

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(4, 1fr)' }, gap: 1.5 }}>
              {providerCards.map((card) => (
                <Box key={card.provider} sx={{ p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2, bgcolor: 'background.paper' }}>
                  <Stack spacing={1}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>{card.label}</Typography>
                      <Chip size="small" label={card.ready ? 'Ready' : card.configured ? 'Check quota' : 'Not configured'} color={card.ready ? 'success' : 'warning'} variant="outlined" />
                      <Button
                        size="small"
                        onClick={() => window.dispatchEvent(new CustomEvent('liga-open-audit-timeline'))}
                        sx={{ ml: 'auto', textTransform: 'none' }}
                      >
                        View timeline
                      </Button>
                    </Box>
                    <Typography variant="body2">Estimated: <strong>{formatUsd(card.estimatedCostUsd)}</strong></Typography>
                    <Typography variant="body2">Known: <strong>{formatUsd(card.knownCostUsd)}</strong></Typography>
                    {card.warnings.length > 0 ? (
                      <Alert severity="warning" sx={{ py: 0 }}>{card.warnings[0]}</Alert>
                    ) : (
                      <Typography variant="caption" color="text.secondary">Quota status may be unknown unless provider reports it.</Typography>
                    )}
                    <Typography variant="caption" color="text.secondary">
                      Recent jobs: {card.recentJobs.length}
                    </Typography>
                  </Stack>
                </Box>
              ))}
            </Box>

            <Divider />

            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}>Recent runs and jobs</Typography>
              {!hasEntries ? (
                <Box sx={{ p: 3, border: '1px dashed', borderColor: 'divider', borderRadius: 2, textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    No usage entries yet. Planning and approval events will appear here before any paid job is launched.
                  </Typography>
                </Box>
              ) : (
                <Stack spacing={1}>
                  {summary.recent_usage_entries.slice(0, 12).map((entry) => (
                    <Box key={entry.usage_id} sx={{ p: 1.25, border: '1px solid', borderColor: 'divider', borderRadius: 2, bgcolor: 'background.paper' }}>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                        <Chip size="small" label={providerLabel(entry.provider)} />
                        <Chip size="small" label={entry.status} variant="outlined" />
                        <Typography variant="body2" sx={{ fontWeight: 700 }}>{usageEntryTitle(entry)}</Typography>
                        <Box sx={{ flex: 1 }} />
                        <Typography variant="body2">Estimate: {formatUsd(entry.estimated_cost_usd)}</Typography>
                        <Typography variant="body2">Approved: {entry.approved ? 'yes' : 'no'}</Typography>
                        <Button
                          size="small"
                          onClick={() => window.dispatchEvent(new CustomEvent('liga-open-audit-timeline'))}
                          sx={{ textTransform: 'none' }}
                        >
                          View timeline
                        </Button>
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        Session {entry.session_id} {entry.run_id ? `· Run ${entry.run_id}` : ''} · Actual provider billing may differ
                      </Typography>
                      {(entry.job_url || entry.artifact_url) && (
                        <Typography variant="caption" sx={{ display: 'block' }}>
                          {entry.job_url && <Link href={entry.job_url} target="_blank" rel="noreferrer">Job link</Link>}
                          {entry.job_url && entry.artifact_url ? ' · ' : ''}
                          {entry.artifact_url && <Link href={entry.artifact_url} target="_blank" rel="noreferrer">Artifact link</Link>}
                        </Typography>
                      )}
                      {(entry.warning || entry.error_summary) && (
                        <Alert severity="warning" sx={{ mt: 1, py: 0 }}>{entry.warning || entry.error_summary}</Alert>
                      )}
                    </Box>
                  ))}
                </Stack>
              )}
            </Box>
          </Stack>
        </DialogContent>
      </Dialog>
    </>
  );
}

function MetricCard({ label, value, helper }: { label: string; value: string; helper: string }) {
  return (
    <Box sx={{ p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2, bgcolor: 'background.paper' }}>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography variant="h6" sx={{ fontWeight: 800 }}>{value}</Typography>
      <Typography variant="caption" color="text.secondary">{helper}</Typography>
    </Box>
  );
}
