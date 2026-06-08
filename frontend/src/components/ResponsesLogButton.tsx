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
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';
import CloseIcon from '@mui/icons-material/Close';

import {
  createResponsesButtonState,
  createResponsesPaginationModel,
  createResponsesPanelModel,
  createResponsesQueryParams,
  type ResponseLogRow,
  type ResponsesPagePayload,
  type ResponsesSummary,
} from '@/lib/responses-log-panel';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';

export default function ResponsesLogButton() {
  const isProcessing = useAgentStore((state) => state.isProcessing);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState<ResponseLogRow[]>([]);
  const [pagePayload, setPagePayload] = useState<ResponsesPagePayload | null>(null);
  const [summary, setSummary] = useState<ResponsesSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [platform, setPlatform] = useState('');
  const [progress, setProgress] = useState('');
  const [model, setModel] = useState('');
  const [jobId, setJobId] = useState('');
  const [q, setQ] = useState('');

  const buttonState = createResponsesButtonState({ isProcessing, summary });
  const filtersActive = Boolean(platform || progress || model.trim() || jobId.trim() || q.trim());
  const panel = useMemo(
    () => createResponsesPanelModel({ rows, error, filtersActive }),
    [rows, error, filtersActive],
  );
  const pagination = useMemo(
    () => createResponsesPaginationModel(pagePayload),
    [pagePayload],
  );

  const loadResponses = useCallback(async (targetPage = page) => {
    setLoading(true);
    setError(null);
    try {
      const params = createResponsesQueryParams({
        page: targetPage,
        pageSize,
        platform,
        progress,
        model,
        jobId,
        q,
      });
      const [rowsResponse, summaryResponse] = await Promise.all([
        apiFetch(`/api/responses?${params.toString()}`),
        apiFetch('/api/responses/summary'),
      ]);
      if (!rowsResponse.ok || !summaryResponse.ok) {
        throw new Error('Failed to load responses.');
      }
      const rowsPayload = await rowsResponse.json();
      const summaryPayload = await summaryResponse.json();
      setRows(Array.isArray(rowsPayload.rows) ? rowsPayload.rows : []);
      setPagePayload(rowsPayload);
      setPage(Number(rowsPayload.page || targetPage));
      setSummary(summaryPayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load responses.');
    } finally {
      setLoading(false);
    }
  }, [jobId, model, page, pageSize, platform, progress, q]);

  const applyFilters = useCallback(() => {
    setPage(1);
    void loadResponses(1);
  }, [loadResponses]);

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
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5, mb: 2 }}>
            <TextField
              label="Search"
              size="small"
              value={q}
              onChange={(event) => setQ(event.target.value)}
            />
            <TextField
              label="Provider"
              select
              size="small"
              value={platform}
              onChange={(event) => setPlatform(event.target.value)}
              sx={{ minWidth: 170 }}
            >
              <MenuItem value="">All providers</MenuItem>
              <MenuItem value="hf-jobs">Hugging Face Jobs</MenuItem>
              <MenuItem value="gcp-vertex">GCP Vertex</MenuItem>
              <MenuItem value="aws-sagemaker">AWS SageMaker</MenuItem>
            </TextField>
            <TextField
              label="Progress"
              select
              size="small"
              value={progress}
              onChange={(event) => setProgress(event.target.value)}
              sx={{ minWidth: 150 }}
            >
              <MenuItem value="">All statuses</MenuItem>
              {['queued', 'running', 'completed', 'failed', 'error', 'cancelled', 'interrupted', 'blocked'].map(
                (status) => (
                  <MenuItem key={status} value={status}>
                    {status}
                  </MenuItem>
                ),
              )}
            </TextField>
            <TextField
              label="Model"
              size="small"
              value={model}
              onChange={(event) => setModel(event.target.value)}
            />
            <TextField
              label="Job / Session"
              size="small"
              value={jobId}
              onChange={(event) => setJobId(event.target.value)}
            />
            <Button variant="contained" size="small" onClick={applyFilters} disabled={loading}>
              Apply
            </Button>
          </Box>

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
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 2 }}>
            <Typography variant="body2" color="text.secondary">
              {pagination.label}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                size="small"
                variant="outlined"
                disabled={!pagination.canGoPrevious || loading}
                onClick={() => {
                  setPage(pagination.previousPage);
                  void loadResponses(pagination.previousPage);
                }}
              >
                Previous
              </Button>
              <Button
                size="small"
                variant="outlined"
                disabled={!pagination.canGoNext || loading}
                onClick={() => {
                  setPage(pagination.nextPage);
                  void loadResponses(pagination.nextPage);
                }}
              >
                Next
              </Button>
            </Box>
          </Box>
        </DialogContent>
      </Dialog>
    </>
  );
}
