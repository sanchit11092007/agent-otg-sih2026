import { useEffect, useRef, useState } from 'react';
import { backendUrl, getSyncContent, streamSyncContent } from './api';

// This is intentionally separate from the presenter UI.  A second device only
// needs this page and a PIN; all generation continues to happen on the host.
export default function ReceiverPage() {
  const [pin, setPin] = useState(() => new URLSearchParams(window.location.search).get('id') || '');
  const [item, setItem] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const streamAbortRef = useRef(null);

  async function loadContent(requestedPin = pin) {
    const cleanPin = String(requestedPin || '').trim();
    if (!cleanPin) {
      setError('Enter the ID shown on the main PC.');
      return;
    }

    streamAbortRef.current?.abort();
    const controller = new AbortController();
    streamAbortRef.current = controller;
    setLoading(true);
    setError('');

    try {
      const response = await getSyncContent(cleanPin);
      setItem(response);
      if (response.status !== 'streaming') return;

      void streamSyncContent(cleanPin, {
        signal: controller.signal,
        onEvent: (event) => {
          if (event.type === 'snapshot') setItem(event);
          if (event.type === 'stage') setItem((current) => current && ({ ...current, stages: [...(current.stages || []), event] }));
          if (event.type === 'token') setItem((current) => current && ({ ...current, content: `${current.content || ''}${event.content || ''}` }));
          if (event.type === 'done') setItem((current) => current && ({ ...current, status: 'completed', artifact: event.artifact || current.artifact }));
          if (event.type === 'error') setItem((current) => current && ({ ...current, status: 'error', error: event.detail || 'The host could not finish this request.' }));
        },
      }).catch((streamError) => {
        if (streamError.name !== 'AbortError') setError('Connection to the host was interrupted. Try the ID again.');
      });
    } catch (requestError) {
      setItem(null);
      setError(requestError.message || 'That ID was not found on the host PC.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const initialPin = new URLSearchParams(window.location.search).get('id');
    // Schedule the optional direct-link lookup after the receiver page mounts.
    const initialLoad = initialPin ? window.setTimeout(() => { void loadContent(initialPin); }, 0) : undefined;
    return () => {
      if (initialLoad) window.clearTimeout(initialLoad);
      streamAbortRef.current?.abort();
    };
    // Only run once: typing an ID must not start a request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="min-h-screen bg-[#100a07] px-4 py-10 text-gray-100">
      <section className="mx-auto max-w-2xl">
        <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-orange-400">Agent OTG receiver</p>
        <h1 className="text-3xl font-bold">Get content from the main PC</h1>
        <p className="mt-2 text-sm text-gray-400">Connect to the same hotspot, enter the ID shown by the presenter, and the result appears here.</p>

        <form className="mt-7 flex gap-2" onSubmit={(event) => { event.preventDefault(); void loadContent(); }}>
          <input
            value={pin}
            onChange={(event) => setPin(event.target.value.replace(/[^a-zA-Z0-9_-]/g, ''))}
            placeholder="Enter ID, e.g. 4821"
            maxLength={20}
            className="min-w-0 flex-1 rounded-xl border border-white/15 bg-black/30 px-4 py-3 font-mono outline-none focus:border-orange-500"
            aria-label="Content ID"
          />
          <button type="submit" disabled={loading} className="rounded-xl bg-orange-500 px-5 py-3 font-semibold text-black disabled:opacity-50">
            {loading ? 'Loading…' : 'Get content'}
          </button>
        </form>

        {error && <p className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">{error}</p>}

        {item && (
          <article className="mt-6 rounded-2xl border border-white/10 bg-black/25 p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-3">
              <span className="font-mono font-bold text-orange-400">ID #{item.sync_id}</span>
              <span className="text-xs text-gray-400">{item.status === 'streaming' ? 'Processing on main PC…' : item.status === 'error' ? 'Failed' : 'Complete'}</span>
            </div>
            <p className="mt-4 text-sm text-gray-400">Request</p>
            <p className="mt-1 whitespace-pre-wrap">{item.query}</p>
            {item.content && <div className="mt-5 whitespace-pre-wrap break-words border-t border-white/10 pt-5 leading-7">{item.content}</div>}
            {item.status === 'streaming' && !item.content && <p className="mt-5 text-sm text-orange-300">The main PC is working on it…</p>}
            {item.error && <p className="mt-5 text-sm text-red-300">{item.error}</p>}
            {item.artifact?.download_url && <a className="mt-5 inline-block rounded-xl bg-orange-500 px-4 py-2 font-semibold text-black" href={backendUrl(item.artifact.download_url)} download>Download {item.artifact.filename || 'file'}</a>}
          </article>
        )}
      </section>
    </main>
  );
}
