/**
 * Top-level React error boundary.
 *
 * Without this, a render-phase exception anywhere in the tree unmounts the
 * whole app and leaves a blank (black) screen — the only recovery being a
 * manual reload. This boundary catches those errors and shows a recoverable
 * fallback instead, so a single bad render can't take down the session UI.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Box, Button, Stack, Typography } from '@mui/material';
import { logger } from '@/utils/logger';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    logger.error('Render error caught by ErrorBoundary:', error, info.componentStack);
  }

  private handleReload = (): void => {
    window.location.reload();
  };

  private handleReset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <Box
        sx={{
          height: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 3,
        }}
      >
        <Stack spacing={2} sx={{ maxWidth: 520, textAlign: 'center' }}>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Something went wrong rendering this view
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            The interface hit an unexpected error. Your session is safe on the server —
            reloading restores it. If it keeps happening, switch models or start a new task.
          </Typography>
          <Typography
            variant="caption"
            component="pre"
            sx={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              opacity: 0.6,
              fontFamily: 'monospace',
              maxHeight: 160,
              overflow: 'auto',
            }}
          >
            {error.message}
          </Typography>
          <Stack direction="row" spacing={1} justifyContent="center">
            <Button variant="contained" onClick={this.handleReload}>
              Reload
            </Button>
            <Button variant="outlined" onClick={this.handleReset}>
              Try again
            </Button>
          </Stack>
        </Stack>
      </Box>
    );
  }
}
