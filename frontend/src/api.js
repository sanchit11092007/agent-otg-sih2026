const configuredBaseUrl = import.meta.env.VITE_API_URL?.replace(/\/$/, '');

// In development Vite proxies /api to FastAPI. Set VITE_API_URL for a deployed
// frontend, for example: VITE_API_URL=https://agent.example.com
const API_BASE_URL = configuredBaseUrl || '/api';

function apiUrl(path) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

async function getErrorMessage(response) {
  try {
    const body = await response.json();
    return body.detail || body.message || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function readWithIdleTimeout(reader, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error('The model stopped sending updates. Please verify Ollama is running and retry.')), timeoutMs);
    reader.read().then(
      (result) => { window.clearTimeout(timer); resolve(result); },
      (error) => { window.clearTimeout(timer); reject(error); },
    );
  });
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 120_000) {
  const controller = new AbortController();
  const inheritedSignal = options.signal;

  if (inheritedSignal?.aborted) {
    throw new DOMException('Generation stopped.', 'AbortError');
  }

  const abortFromCaller = () => controller.abort(inheritedSignal?.reason);
  const timer = window.setTimeout(() => controller.abort(new DOMException('Request timed out', 'TimeoutError')), timeoutMs);

  inheritedSignal?.addEventListener('abort', abortFromCaller, { once: true });
  try {
    return await fetch(url, { ...options, cache: 'no-store', signal: controller.signal });
  } catch (error) {
    if (inheritedSignal?.aborted) {
      throw new DOMException('Generation stopped.', 'AbortError');
    }
    if (controller.signal.aborted) {
      throw new Error('The local server took too long to respond. Please check the backend and try again.', { cause: error });
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
    inheritedSignal?.removeEventListener('abort', abortFromCaller);
  }
}

async function requestJson(path, options = {}, { timeoutMs = 120_000, retry = false } = {}) {
  const attempts = retry ? 2 : 1;
  let lastError;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (options.signal?.aborted) {
      throw new DOMException('Generation stopped.', 'AbortError');
    }
    try {
      const response = await fetchWithTimeout(apiUrl(path), options, timeoutMs);
      if (!response.ok) throw new Error(await getErrorMessage(response));
      return response.json();
    } catch (error) {
      if (options.signal?.aborted || error.name === 'AbortError') {
        throw new DOMException('Generation stopped.', 'AbortError');
      }
      lastError = error;
      if (attempt < attempts - 1) await wait(300);
    }
  }
  throw lastError;
}

export function backendUrl(path) {
  if (!path) return API_BASE_URL;
  return apiUrl(path);
}

export function getHealth() {
  return requestJson('/health', {}, { timeoutMs: 5_000, retry: true });
}

export function getCapabilities() {
  return requestJson('/capabilities', {}, { timeoutMs: 15_000 });
}

export function resetConversation() {
  return requestJson('/reset', { method: 'POST' }, { timeoutMs: 15_000 });
}

export function getSessions() {
  return requestJson('/sessions', {}, { timeoutMs: 15_000, retry: true });
}

export function getSessionMessages(sessionName) {
  return requestJson(`/sessions/${encodeURIComponent(sessionName)}`, {}, { timeoutMs: 20_000, retry: true });
}

export function ingestKnowledgeBase(paths) {
  return requestJson('/knowledge-base/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths, replace_existing: false }),
  }, { timeoutMs: 180_000 });
}

export async function uploadKnowledgeBase(files, { signal } = {}) {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  const response = await fetchWithTimeout(apiUrl('/knowledge-base/upload'), {
    method: 'POST',
    body: formData,
    signal,
  }, 180_000);
  if (!response.ok) throw new Error(await getErrorMessage(response));
  return response.json();
}

export function clearKnowledgeBase() {
  return requestJson('/knowledge-base/clear', { method: 'POST' }, { timeoutMs: 30_000 });
}


export function askAboutImage(query, image_b64, { signal } = {}) {
  return requestJson('/ask/image', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, image_b64 }),
    signal,
  }, { timeoutMs: 300_000 });
}

/**
 * Consume FastAPI's newline-delimited JSON stream. The backend emits stages,
 * plans, tokens and a final done event, so the caller can update the chat UI
 * while the model is working instead of waiting for a single final response.
 */
export async function streamQuestion(query, { signal, onEvent }) {
  const response = await fetchWithTimeout(apiUrl('/ask/stream'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal,
  }, 120_000);

  if (!response.ok) throw new Error(await getErrorMessage(response));
  if (!response.body) throw new Error('The backend returned an empty response stream.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const onAbort = () => {
    try {
      reader.cancel(new DOMException('Generation stopped.', 'AbortError'));
    } catch {
      // Stream may already be closed
    }
  };

  if (signal?.aborted) {
    onAbort();
    throw new DOMException('Generation stopped.', 'AbortError');
  }

  signal?.addEventListener('abort', onAbort, { once: true });

  const processLine = (line) => {
    if (!line.trim()) return;
    try {
      onEvent(JSON.parse(line));
    } catch {
      // A malformed event should not make an otherwise usable answer disappear.
    }
  };

  try {
    while (true) {
      if (signal?.aborted) {
        throw new DOMException('Generation stopped.', 'AbortError');
      }
      const { done, value } = await readWithIdleTimeout(reader, 600_000);
      if (signal?.aborted) {
        throw new DOMException('Generation stopped.', 'AbortError');
      }
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || '';
      lines.forEach(processLine);
      if (done) break;
    }
    processLine(buffer);
  } finally {
    signal?.removeEventListener('abort', onAbort);
    try {
      await reader.cancel();
    } catch {
      // The stream may already have completed or have been aborted.
    }
    reader.releaseLock();
  }
}

/**
 * Get host network info (Hotspot and LAN IP addresses)
 */
export function getNetworkInfo() {
  return requestJson('/network/info', {}, { timeoutMs: 5_000, retry: true });
}

/**
 * Fetch a synced broadcast result by short PIN / ID
 */
export function getSyncContent(syncId) {
  if (!syncId) throw new Error('Sync ID is required.');
  return requestJson(`/sync/content/${encodeURIComponent(String(syncId).trim())}`, {}, { timeoutMs: 15_000 });
}

/**
 * Retrieve recent shared broadcasts
 */
export function getRecentSyncs(limit = 15) {
  return requestJson(`/sync/recent?limit=${limit}`, {}, { timeoutMs: 10_000, retry: true });
}

/**
 * Get the latest active or completed sync item
 */
export function getLatestSync() {
  return requestJson('/sync/latest', {}, { timeoutMs: 8_000, retry: true });
}

/**
 * Submit a question from a remote/receiver device to be processed on the main server
 */
export function submitRemoteTask(query, { syncId, mode = 'agent' } = {}) {
  return requestJson('/sync/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, sync_id: syncId, mode }),
  }, { timeoutMs: 30_000 });
}

/**
 * Stream real-time progress for a specific Sync ID
 */
export async function streamSyncContent(syncId, { signal, onEvent }) {
  if (!syncId) throw new Error('Sync ID is required.');
  const response = await fetchWithTimeout(apiUrl(`/sync/stream/${encodeURIComponent(String(syncId).trim())}`), {
    method: 'GET',
    signal,
  }, 120_000);

  if (!response.ok) throw new Error(await getErrorMessage(response));
  if (!response.body) throw new Error('The backend returned an empty response stream.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const onAbort = () => {
    try {
      reader.cancel(new DOMException('Stream stopped.', 'AbortError'));
    } catch {}
  };

  if (signal?.aborted) {
    onAbort();
    throw new DOMException('Stream stopped.', 'AbortError');
  }

  signal?.addEventListener('abort', onAbort, { once: true });

  const processLine = (line) => {
    if (!line.trim()) return;
    try {
      const parsed = JSON.parse(line);
      if (parsed.type !== 'ping') {
        onEvent(parsed);
      }
    } catch {}
  };

  try {
    while (true) {
      if (signal?.aborted) throw new DOMException('Stream stopped.', 'AbortError');
      const { done, value } = await readWithIdleTimeout(reader, 600_000);
      if (signal?.aborted) throw new DOMException('Stream stopped.', 'AbortError');
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || '';
      lines.forEach(processLine);
      if (done) break;
    }
    processLine(buffer);
  } finally {
    signal?.removeEventListener('abort', onAbort);
    try {
      await reader.cancel();
    } catch {}
    reader.releaseLock();
  }
}

