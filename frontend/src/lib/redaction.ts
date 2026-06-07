const REDACTED = '[REDACTED]';

const SECRET_KEY_RE = /token|secret|password|credential|api[_-]?key|access[_-]?key|private[_-]?key|session[_-]?token|authorization|mongodb_uri|google_application_credentials/i;
const TOKEN_PATTERNS: RegExp[] = [
  /hf_[A-Za-z0-9]{20,}/g,
  /sk-ant-[A-Za-z0-9_-]{20,}/g,
  /sk-proj-[A-Za-z0-9_-]{20,}/g,
  /sk-(?!ant-|proj-)[A-Za-z0-9_-]{32,}/g,
  /gh[pousr]_[A-Za-z0-9]{30,}/g,
  /github_pat_[A-Za-z0-9_]{30,}/g,
  /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g,
];
const BEARER_RE = /\b(bearer)\s+[A-Za-z0-9_.=:/+-]{12,}/gi;
const ENV_ASSIGNMENT_RE = /\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|CREDENTIALS?|API_KEY|PRIVATE_KEY|ACCESS_KEY|SESSION_TOKEN|MONGODB_URI)[A-Z0-9_]*)\s*([=:])\s*([^ \t\r\n"']+)/gim;
const PRIVATE_KEY_RE = /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g;
const MONGO_URI_RE = /\b(mongodb(?:\+srv)?:\/\/)([^/@\s:]+):([^/@\s]+)@([^\s/?#]+)([^\s]*)/gi;
const SIGNED_URL_KEYS_RE = /[?&](X-Amz-Signature|X-Amz-Credential|X-Amz-Security-Token|GoogleAccessId|Signature|X-Goog-Signature|X-Goog-Credential)=/i;
const CREDENTIAL_PATH_RE = /(^|[\s=:])(\/[^\s]*?(?:credentials?|service[-_]?account)[^\s]*?\.json)\b/gi;

export function redactText(value: string): string {
  if (!value) return value;
  let out = value
    .replace(PRIVATE_KEY_RE, REDACTED)
    .replace(CREDENTIAL_PATH_RE, (_match, prefix: string) => `${prefix}${REDACTED}`)
    .replace(MONGO_URI_RE, `$1${REDACTED}@$4$5`)
    .replace(BEARER_RE, (_match, scheme: string) => `${scheme} ${REDACTED}`)
    .replace(ENV_ASSIGNMENT_RE, (_match, key: string, sep: string) => `${key}${sep}${REDACTED}`);

  for (const pattern of TOKEN_PATTERNS) {
    out = out.replace(pattern, REDACTED);
  }
  if (SIGNED_URL_KEYS_RE.test(out)) {
    out = out.replace(/\?[^ \t\r\n)]+/g, `?${REDACTED}`);
  }
  return out;
}

export function redactJsonLike<T>(value: T): T {
  if (typeof value === 'string') return redactText(value) as T;
  if (Array.isArray(value)) return value.map((item) => redactJsonLike(item)) as T;
  if (!value || typeof value !== 'object') return value;

  const clean: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (/^authorization$/i.test(key) && typeof entry === 'string') {
      clean[key] = redactText(entry);
    } else {
      clean[key] = SECRET_KEY_RE.test(key) ? REDACTED : redactJsonLike(entry);
    }
  }
  return clean as T;
}

export function containsSecretLikeValue(value: unknown): boolean {
  return JSON.stringify(redactJsonLike(value)) !== JSON.stringify(value);
}

export function redactedJsonString(value: unknown, spacing = 2): string {
  return JSON.stringify(redactJsonLike(value), null, spacing);
}
