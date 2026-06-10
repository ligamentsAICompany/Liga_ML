interface PanelSectionLike {
  content: string;
  language: string;
}

interface PanelDataLike {
  title: string;
  script?: PanelSectionLike;
  output?: PanelSectionLike;
  input?: PanelSectionLike;
  parameters?: Record<string, unknown>;
}

type PanelViewLike = 'script' | 'output';

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'undefined';
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(',')}]`;
  }
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
    .join(',')}}`;
}

function panelKey(
  data: PanelDataLike | null,
  view: PanelViewLike,
  editable: boolean,
): string {
  return stableStringify({ data, view, editable });
}

export function isSamePanelState(
  currentData: PanelDataLike | null,
  currentView: PanelViewLike,
  currentEditable: boolean,
  nextData: PanelDataLike,
  nextView?: PanelViewLike,
  nextEditable?: boolean,
): boolean {
  const resolvedNextView = nextView ?? (nextData.script ? 'script' : 'output');
  const resolvedNextEditable = nextEditable ?? false;
  return (
    panelKey(currentData, currentView, currentEditable)
    === panelKey(nextData, resolvedNextView, resolvedNextEditable)
  );
}
