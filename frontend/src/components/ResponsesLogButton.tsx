import { useCallback, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';
import CloseIcon from '@mui/icons-material/Close';

import {
  createResponsesButtonState,
  createResponsesPanelModel,
  type ResponseLogRow,
  type ResponsesSummary,
} from '@/lib/responses-log-panel';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';

export default function ResponsesLogButton() {
  const isProcessing = useAgentStore((state) => state.isProcessing);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState<ResponseLogRow[]>([]);
  const [summary, setSummary] = useState<ResponsesSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const buttonState = createResponsesButtonState({ isProcessing, summary });
  const panel = useMemo(
    () => createResponsesPanelModel({ rows, error }),
    [rows, error],
  );

  const loadResponses = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [rowsResponse, summaryResponse] = await Promise.all([
        apiFetch('/api/responses'),
        apiFetch('/api/responses/summary'),
      ]);
      if (!rowsResponse.ok || !summaryResponse.ok) {
        throw new Error('Failed to load responses.');
      }
      const rowsPayload = await rowsResponse.json();
      const summaryPayload = await summaryResponse.json();
      setRows(Array.isArray(rowsPayload.rows) ? rowsPayload.rows : []);
      setSummary(summaryPayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load responses.');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleOpen = useCallback(() => {
    setOpen(true);
    void loadResponses();
  }, [loadResponses]);

  return (
    <>
      <Button
        size="small"
        variant="outlined"
        startIcon={<FormatListBulletedIcon />}
        onClick={handleOpen}
        disabled={buttonState.disabled}
        aria-label="Open Responses log"
        sx={{
          textTransform: 'none',
          borderColor: 'var(--border-hover)',
          color: 'var(--text)',
          bgcolor: 'rgba(255,255,255,0.7)',
          '&:hover': { borderColor: 'var(--accent-green)', bgcolor: 'var(--hover-bg)' },
        }}
      >
        {buttonState.label}
      </Button>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="xl" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
            Responses Log
            <Chip
              size="small"
              label={`Batch ${summary?.batch_number || 1}`}
              sx={{ bgcolor: 'var(--accent-yellow-weak)', color: 'var(--text)' }}
            />
          </Box>
          <IconButton aria-label="Close Responses log" onClick={() => setOpen(false)} size="small">
            <CloseIcon fontSize="small" />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {panel.errorMessage ? (
            <Alert severity="error" sx={{ mb: 2 }}>
              {panel.errorMessage}
            </Alert>
          ) : null}

          {loading ? (
            <Box sx={{ py: 6, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress size={28} />
            </Box>
          ) : panel.rows.length === 0 ? (
            <Box
              sx={{
                py: 7,
                px: 2,
                textAlign: 'center',
                border: '1px dashed var(--border-hover)',
                borderRadius: 2,
                bgcolor: 'var(--panel)',
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                {panel.emptyStateTitle}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {panel.emptyStateDescription}
              </Typography>
            </Box>
          ) : (
            <TableContainer sx={{ maxHeight: '70vh', border: '1px solid var(--border)', borderRadius: 2 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    {panel.columns.map((column) => (
                      <TableCell key={column.key} sx={{ fontWeight: 700, whiteSpace: 'nowrap' }}>
                        {column.label}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {panel.rows.map((row) => (
                    <TableRow key={`${row.raw.session_id}-${row.raw.job_id}-${row.raw.actual_sequence_number}`} hover>
                      <TableCell>{row.cells.session}</TableCell>
                      <TableCell>{row.cells.model}</TableCell>
                      <TableCell>{row.cells.platform}</TableCell>
                      <TableCell>{row.cells.runType}</TableCell>
                      <TableCell>{row.cells.storage}</TableCell>
                      <TableCell>{row.cells.progress}</TableCell>
                      <TableCell sx={{ maxWidth: 220, wordBreak: 'break-all' }}>{row.cells.jobId}</TableCell>
                      <TableCell sx={{ maxWidth: 300, wordBreak: 'break-all' }}>{row.cells.result}</TableCell>
                      <TableCell>{row.cells.createdAt}</TableCell>
                      <TableCell>{row.cells.completedAt}</TableCell>
                      <TableCell>{row.cells.shortSessionId}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
