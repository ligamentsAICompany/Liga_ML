export interface TrainingResult {
  status?: string;
  provider?: 'gcp-vertex' | 'hf-jobs' | string;
  outputPolicy?: string;
  finalModelUrl?: string;
  hubModelId?: string;
  gcsOutputDir?: string;
  evalResult?: Record<string, unknown> | null;
  resultFile?: string;
}

const MARKERS = {
  status: 'LIGA_TRAINING_STATUS',
  provider: 'LIGA_PROVIDER',
  outputPolicy: 'LIGA_OUTPUT_POLICY',
  finalModelUrl: 'LIGA_FINAL_MODEL_URL',
  hubModelId: 'LIGA_HUB_MODEL_ID',
  gcsOutputDir: 'LIGA_GCS_OUTPUT_DIR',
  evalResult: 'LIGA_EVAL_RESULT_JSON',
  resultFile: 'LIGA_RESULT_FILE',
} as const;

const TRAILING_URL_PUNCTUATION = /[.,;:)\]}]+$/;

function markerValue(output: string, marker: string): string | undefined {
  const pattern = new RegExp(`${marker}=([^\\r\\n]*)`, 'g');
  let match: RegExpExecArray | null;
  let value: string | undefined;
  while ((match = pattern.exec(output)) !== null) {
    value = match[1]?.trim();
  }
  return value || undefined;
}

function cleanUrl(value: string | undefined): string | undefined {
  return value?.replace(TRAILING_URL_PUNCTUATION, '');
}

function parseEvalResult(value: string | undefined): Record<string, unknown> | null | undefined {
  if (value === undefined) return undefined;
  try {
    const parsed: unknown = JSON.parse(value);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
}

export function parseLigaTrainingResult(output: string | undefined): TrainingResult | null {
  if (!output) return null;

  const hasMarker = Object.values(MARKERS).some((marker) => output.includes(`${marker}=`));
  if (!hasMarker) return null;

  const result: TrainingResult = {};
  const status = markerValue(output, MARKERS.status);
  const provider = markerValue(output, MARKERS.provider);
  const outputPolicy = markerValue(output, MARKERS.outputPolicy);
  const finalModelUrl = cleanUrl(markerValue(output, MARKERS.finalModelUrl));
  const hubModelId = markerValue(output, MARKERS.hubModelId);
  const gcsOutputDir = markerValue(output, MARKERS.gcsOutputDir);
  const evalResult = parseEvalResult(markerValue(output, MARKERS.evalResult));
  const resultFile = markerValue(output, MARKERS.resultFile);

  if (status) result.status = status;
  if (provider) result.provider = provider;
  if (outputPolicy) result.outputPolicy = outputPolicy;
  if (finalModelUrl) result.finalModelUrl = finalModelUrl;
  if (hubModelId) result.hubModelId = hubModelId;
  if (gcsOutputDir) result.gcsOutputDir = gcsOutputDir;
  if (evalResult !== undefined) result.evalResult = evalResult;
  if (resultFile) result.resultFile = resultFile;

  return Object.keys(result).length > 0 ? result : null;
}
