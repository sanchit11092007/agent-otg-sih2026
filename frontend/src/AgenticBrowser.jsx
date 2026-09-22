import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowRight, Cast, Check, ChevronDown, ChevronsLeft, Copy, Database,
  Download, ExternalLink, FileCode, FileSpreadsheet, FileText, Keyboard, Laptop, LoaderCircle,
  MessageSquare, Paperclip, Pencil, Pin, Plus, Radio, RefreshCw, RotateCw, Search, Send, Settings,
  Share2, Smartphone, Square, Trash2, Tv, Upload, Wifi, WifiOff, X,
} from 'lucide-react';
import {
  askAboutImage, backendUrl, clearKnowledgeBase, getHealth, getLatestSync, getNetworkInfo,
  getRecentSyncs, getSessionMessages, getSessions, getSyncContent, ingestKnowledgeBase,
  resetConversation, streamQuestion, streamSyncContent, submitRemoteTask, uploadKnowledgeBase,
} from './api';


const CHAT_STORAGE_KEY = 'agent-otg-client-chats-v1';
const ACTIVE_CHAT_STORAGE_KEY = 'agent-otg-active-chat-v1';
const TITLE_STORAGE_KEY = 'agent-otg-history-titles-v1';
const HIDDEN_SESSIONS_STORAGE_KEY = 'agent-otg-hidden-sessions-v1';
const SESSION_TITLE_CACHE_KEY = 'agent-otg-session-title-cache-v4';
const PINNED_HISTORY_KEY = 'agent-otg-pinned-history-v1';
const RAG_MEMORY_KEY = 'agent-otg-rag-memory-v1';
const AUTO_NEW_CHAT_STORAGE_KEY = 'agent-otg-auto-new-chat-on-start-v1';
const HISTORY_PAGE_SIZE = 8;
const HEADING_PREFIX = 'Ready to Ask ';
const HEADING_HIGHLIGHT = 'Off The Grid';
const HEADING_SUFFIX = '?';

const DOC_TYPES = {
  pdf: { label: 'PDF Document', noun: 'PDF document', icon: FileText, accent: 'text-red-400' },
  excel: { label: 'Excel Sheet', noun: 'Excel spreadsheet', icon: FileSpreadsheet, accent: 'text-emerald-400' },
  word: { label: 'Word Doc', noun: 'Word document', icon: FileCode, accent: 'text-blue-400' },
};

function readStoredJson(key, fallback) {
  try {
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function writeStoredJson(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Private-mode and quota errors must not prevent the chat from working.
  }
}

function createId(prefix = 'chat') {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function createChat() {
  const now = new Date().toISOString();
  return { id: createId(), title: 'New chat', titleEdited: false, createdAt: now, updatedAt: now, messages: [], isRagEnabled: false };
}

function deriveTitle(text) {
  const source = String(text || '').replace(/\s+/g, ' ').trim();
  if (!source) return 'New chat';
  // History should be an honest, predictable preview of the first prompt.
  // Keep punctuation with the words and always use the requested ellipsis.
  const words = source.split(/\s+/).slice(0, 4);
  return `${words.join(' ')}.......`;
}

function isModelConnectionError(value) {
  return /error during streaming|winerror 10061|connection refused|localhost['"]?,?\s*port['"]?\s*[:=]?\s*11434|ollama/i.test(String(value || ''));
}

function friendlyErrorMessage(value) {
  const raw = String(value || 'The request could not be completed.');
  if (isModelConnectionError(raw)) {
    return 'The backend is online, but Ollama is not running on http://localhost:11434. Start Ollama and make sure the required Qwen model is installed, then try again.';
  }
  return raw.replace(/^\[Error during streaming:\s*|\]$/g, '');
}

function toUiMessages(messages = []) {
  return messages.map((message) => ({
    id: createId('message'),
    role: message.role === 'user' ? 'user' : 'assistant',
    content: message.content || '',
    createdAt: message.created_at || new Date().toISOString(),
    stages: [],
  }));
}

function isImageFile(file) {
  return Boolean(file?.type?.startsWith('image/') || /\.(png|jpe?g|webp|gif|bmp|svg|tiff?)$/i.test(file?.name || ''));
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.onload = () => resolve(String(reader.result).split(',').pop());
    reader.readAsDataURL(file);
  });
}

function isReferenceRequest(prompt) {
  return /\b(?:above|previous|earlier|same|that|this|it|them|those)\b/i.test(prompt);
}

function conversationContext(messages = []) {
  const recent = messages
    .filter((message) => message.role === 'user' && message.content?.trim())
    .slice(-5)
    .map((message) => message.content.trim())
    .join('\n');
  return recent.slice(0, 8_000);
}

function attachmentContext(selectedAttachments, ragMemory) {
  const selectedKeys = new Set(selectedAttachments.map((file) => `${file.name}-${file.size}-${file.lastModified}`));
  const relevantEntries = selectedKeys.size
    ? ragMemory.filter((entry) => selectedKeys.has(entry.key))
    : ragMemory.slice(-6);
  return relevantEntries
    .map((entry) => `File: ${entry.name}\n${entry.text}`)
    .join('\n\n---\n\n')
    .slice(0, 16_000);
}

function buildSubmissionPrompt(prompt, activeFilter, mode, messages = [], fileContext = '', useKnowledgeBase = false) {
  const cleanPrompt = prompt.trim();
  if (mode === 'chief') return `/complex ${cleanPrompt}`;
  const context = isReferenceRequest(cleanPrompt) ? conversationContext(messages) : '';
  let documentInstruction = cleanPrompt;
  if (activeFilter === 'excel') {
    documentInstruction = `Create an Excel spreadsheet (.xlsx) about ${cleanPrompt}`;
  } else if (activeFilter === 'word') {
    documentInstruction = `Create a Word document (.docx) about ${cleanPrompt}`;
  } else if (activeFilter === 'pdf') {
    documentInstruction = `Create a PDF document (.pdf) about ${cleanPrompt}`;
  }
  const parts = [documentInstruction];
  if (context) parts.push(`Earlier user context:\n${context}`);
  if (fileContext) parts.push(`Selected file context:\n${fileContext}`);
  if (useKnowledgeBase && !activeFilter) {
    parts.push(
      'Use the uploaded knowledge base as the primary source. '
      + 'Search indexed documents, ground every claim in retrieved context, '
      + 'and cite the source file names you used.',
    );
  }
  return parts.join('\n\n');
}

function isReadableTextFile(file) {
  return /^text\//i.test(file.type) || /\.(?:txt|md|csv|json|js|jsx|ts|tsx|py|java|c|cpp|cs|html|css|xml|yaml|yml|log)$/i.test(file.name);
}

function describeStage(stage) {
  const key = String(stage?.label || '').toLowerCase();
  const labels = {
    understanding: 'Request analysis',
    planning: 'Execution plan',
    plan: 'Execution plan',
    routing: 'Model routing',
    attachments: 'Knowledge retrieval',
    availability: 'Availability check',
    vision: 'Vision analysis',
    working: 'Tool orchestration',
    generating: 'Response generation',
    done: 'Completed',
  };
  return { title: labels[key] || 'Processing', detail: stage?.detail || 'Preparing the response…' };
}

function formatDate(value) {
  const date = new Date(value || '');
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function CopyButton({ value, copiedId, copyId, variant = 'assistant', onCopy }) {
  const copied = copiedId === copyId;
  const copy = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(value);
        } catch {
          const textarea = document.createElement('textarea');
          textarea.value = value;
          textarea.style.position = 'fixed';
          textarea.style.opacity = '0';
          document.body.appendChild(textarea);
          textarea.select();
          const ok = document.execCommand('copy');
          textarea.remove();
          if (!ok) throw new Error('Copy unavailable');
        }
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = value;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        const ok = document.execCommand('copy');
        textarea.remove();
        if (!ok) throw new Error('Copy unavailable');
      }
      onCopy?.(copyId);
    } catch {
      // Ignored
    }
  };

  let styleClasses = 'bg-white/5 border border-white/10 px-2.5 py-1 text-xs text-gray-300 hover:bg-white/10 hover:text-white rounded-lg shadow-sm';
  if (variant === 'user') {
    styleClasses = 'bg-black/15 hover:bg-black/25 text-[#3d1600] border border-black/10 px-2.5 py-1 text-xs font-semibold rounded-lg shadow-sm transition-all';
  } else if (variant === 'code') {
    styleClasses = 'text-gray-400 hover:text-gray-200 hover:bg-white/10 px-2 py-1 text-xs rounded-md transition-colors';
  } else if (variant === 'assistant') {
    styleClasses = 'bg-white/[0.04] border border-white/10 px-2.5 py-1 text-xs font-medium text-gray-300 hover:bg-white/10 hover:text-white rounded-lg shadow-sm transition-colors';
  }

  return (
    <button
      type="button"
      onClick={copy}
      className={`inline-flex items-center gap-1.5 ${styleClasses}`}
      title="Copy to clipboard"
    >
      {copied ? <Check className={`h-3.5 w-3.5 ${variant === 'user' ? 'text-[#3d1600]' : 'text-emerald-400'}`} /> : <Copy className="h-3.5 w-3.5" />}
      <span>{copied ? 'Copied!' : 'Copy'}</span>
    </button>
  );
}

function InlineText({ children }) {
  const parts = String(children || '').split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith('`') && part.endsWith('`')) return <code key={index} className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[0.9em] text-orange-100">{part.slice(1, -1)}</code>;
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={index} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
    return part;
  });
}

function Prose({ value }) {
  const lines = String(value || '').split('\n');
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) {
      index += 1;
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.+)$/) || line.match(/^(.{3,80}):$/);
    if (heading) {
      blocks.push(<h3 key={`heading-${index}`} className="pt-1 text-base font-semibold text-white"><InlineText>{heading[1]}</InlineText></h3>);
      index += 1;
      continue;
    }
    const isBullet = /^[-*•]\s+/.test(line);
    const isNumbered = /^\d+[.)]\s+/.test(line);
    if (isBullet || isNumbered) {
      const items = [];
      const matcher = isNumbered ? /^\d+[.)]\s+(.+)$/ : /^[-*•]\s+(.+)$/;
      while (index < lines.length && matcher.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(matcher, '$1'));
        index += 1;
      }
      const List = isNumbered ? 'ol' : 'ul';
      blocks.push(<List key={`list-${index}`} className={`${isNumbered ? 'list-decimal' : 'list-disc'} space-y-1.5 pl-5 marker:text-[#ff6700]`}>{items.map((item, itemIndex) => <li key={itemIndex}><InlineText>{item}</InlineText></li>)}</List>);
      continue;
    }
    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !/^#{1,6}\s+/.test(lines[index]) && !/^[-*•]\s+/.test(lines[index]) && !/^\d+[.)]\s+/.test(lines[index])) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push(<p key={`paragraph-${index}`}><InlineText>{paragraph.join(' ')}</InlineText></p>);
  }
  return blocks;
}

function FormattedAnswer({ content, copiedId, onCopied, messageId }) {
  if (!content) return null;
  const sections = content.split(/(```[\s\S]*?```)/g).filter(Boolean);
  return (
    <div className="space-y-3 text-[15px] leading-7 text-gray-200">
      {sections.map((section, index) => {
        if (!section.startsWith('```')) return <Prose key={`${messageId}-text-${index}`} value={section} />;
        const match = section.match(/^```([^\n]*)\n?([\s\S]*?)```$/);
        const language = match?.[1]?.trim() || 'code';
        const code = match?.[2] || section.replace(/^```|```$/g, '');
        const copyId = `${messageId}-code-${index}`;
        return (
          <div key={copyId} className="overflow-hidden rounded-xl border border-white/10 bg-[#090705] shadow-inner">
            <div className="flex items-center justify-between border-b border-white/5 px-3 py-2 text-xs text-gray-500">
              <span className="font-mono">{language}</span>
              <CopyButton value={code} copiedId={copiedId} copyId={copyId} variant="code" onCopy={onCopied} />
            </div>
            <pre className="overflow-x-auto p-4 text-[13px] leading-6 text-orange-100"><code>{code}</code></pre>
          </div>
        );
      })}
    </div>
  );
}

function UserMessage({ message, copiedId, onCopied }) {
  return (
    <article className="group relative ml-auto max-w-[85%] rounded-2xl rounded-br-md bg-[#ff6700] px-4 py-3 pr-24 text-[15px] leading-6 text-[#210b01] shadow-lg shadow-[#ff6700]/10">
      {message.documentType && <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-[#7a2800]">{DOC_TYPES[message.documentType]?.label}</div>}
      <div className="whitespace-pre-wrap break-words">{message.content}</div>
      {message.attachments?.length > 0 && <div className="mt-2 flex flex-wrap gap-1.5">{message.attachments.map((file) => <span key={`${message.id}-${file.name}`} className="inline-flex max-w-full items-center gap-1 rounded-md bg-black/10 px-2 py-1 text-xs text-[#5e2000]"><Paperclip className="h-3 w-3" /><span className="truncate">{file.name}</span></span>)}</div>}
      <div className="absolute right-2 top-2 opacity-80 transition-opacity hover:opacity-100 focus-within:opacity-100"><CopyButton value={message.content} copiedId={copiedId} copyId={message.id} variant="user" onCopy={onCopied} /></div>
    </article>
  );
}

function AssistantMessage({ message, copiedId, onCopied, onShareSync }) {
  const stages = message.stages || [];
  const sources = message.meta?.sources || [];
  const artifact = message.meta?.artifact;
  const syncId = message.syncId;

  return (
    <article className="group rounded-2xl border border-white/[0.07] bg-[#17100c]/80 p-4 shadow-xl shadow-black/10 sm:p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium text-[#ff8b46]">
          <img src="/logo.png" alt="" className="h-5 w-5 object-contain" />
          <span>Agent OTG</span>
          {message.meta?.modelUsed && <span className="rounded-full bg-[#ff6700]/10 px-2 py-0.5 text-[11px] text-orange-200">{message.meta.modelUsed}</span>}
          {syncId && (
            <button
              type="button"
              onClick={() => onShareSync?.(syncId)}
              className="inline-flex items-center gap-1.5 rounded-full border border-[#ff6700]/30 bg-[#ff6700]/10 px-2.5 py-0.5 text-[11px] font-mono font-medium text-[#ff944d] transition-all hover:bg-[#ff6700]/25 hover:scale-105"
              title="Device Sync PIN — Click to share or view in Device Hub"
            >
              <Radio className="h-3 w-3 animate-pulse text-[#ff6700]" />
              <span>PIN #{syncId}</span>
            </button>
          )}
        </div>
        {message.content && <CopyButton value={message.content} copiedId={copiedId} copyId={message.id} variant="assistant" onCopy={onCopied} />}
      </div>


      {stages.length > 0 && (
        <div className="mb-3 flex min-w-0 items-center gap-2 rounded-xl border border-[#ff6700]/15 bg-black/25 px-3 py-2.5 text-xs text-orange-100" role="status" aria-live="polite">
          {message.streaming ? <LoaderCircle className="h-3.5 w-3.5 shrink-0 animate-spin text-[#ff6700]" /> : message.error ? <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-red-400" /> : <Check className="h-3.5 w-3.5 shrink-0 text-emerald-400" />}
          <span className="truncate"><span className="font-medium text-orange-50">{describeStage(stages.at(-1)).title}</span> · {describeStage(stages.at(-1)).detail}</span>
        </div>
      )}

      {message.error && <div className="mb-3 flex gap-2 rounded-xl border border-red-400/20 bg-red-500/10 p-3 text-sm text-red-200"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /><span>{message.error}</span></div>}
      <FormattedAnswer content={message.content} copiedId={copiedId} onCopied={onCopied} messageId={message.id} />

      {artifact?.download_url && <a href={backendUrl(artifact.download_url)} className="mt-4 inline-flex items-center gap-2 rounded-xl bg-[#ff6700]/15 px-3 py-2 text-sm font-medium text-[#ffad76] transition-colors hover:bg-[#ff6700]/25"><Download className="h-4 w-4" />Download {artifact.filename || 'generated file'}</a>}

      {sources.length > 0 && (
        <div className="mt-4 border-t border-white/5 pt-3">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-gray-500"><Database className="h-3.5 w-3.5" />Sources</div>
          <div className="flex flex-wrap gap-2">
            {sources.map((source, index) => {
              const label = typeof source === 'string' ? source : source.source || source.filename || source.metadata?.source || `Source ${index + 1}`;
              return <span key={`${message.id}-source-${index}`} className="max-w-full truncate rounded-lg bg-white/5 px-2.5 py-1 text-xs text-gray-400">{label}</span>;
            })}
          </div>
        </div>
      )}
    </article>
  );
}

function StatusPill({ apiStatus, isOnline }) {
  const offline = !isOnline || apiStatus === 'offline';
  const degraded = isOnline && apiStatus === 'degraded';
  const color = offline ? 'text-red-400' : degraded ? 'text-amber-400' : 'text-emerald-400';
  const label = !isOnline ? 'Offline' : offline ? 'Backend offline' : degraded ? 'Model offline' : 'Backend online';
  return (
    <div className="flex flex-col items-center gap-0.5 rounded-lg border border-white/5 bg-black/40 px-3 py-1.5 backdrop-blur-sm" title={label}>
      <div className="flex items-center gap-2 text-xs font-mono font-medium">
        <Activity className={`h-3.5 w-3.5 ${color}`} />
        <span className={offline ? 'text-red-200' : degraded ? 'text-amber-200' : 'text-emerald-200'}>{degraded ? 'Model unavailable' : label}</span>
      </div>
    </div>
  );
}

function DeviceSyncModal({ isOpen, onClose, initialSyncId = '', onSelectChat }) {
  const [activeTab, setActiveTab] = useState(initialSyncId ? 'receiver' : 'host');
  const [syncIdInput, setSyncIdInput] = useState(initialSyncId);
  const [networkInfo, setNetworkInfo] = useState(null);
  const [recentSyncs, setRecentSyncs] = useState([]);
  const [isLoadingNet, setIsLoadingNet] = useState(false);
  const [activeContent, setActiveContent] = useState(null);
  const [isSyncLoading, setIsSyncLoading] = useState(false);
  const [syncError, setSyncError] = useState('');
  const [copiedKey, setCopiedKey] = useState('');
  const [autoFollow, setAutoFollow] = useState(false);
  const [remotePrompt, setRemotePrompt] = useState('');
  const [isSubmittingRemote, setIsSubmittingRemote] = useState(false);
  const [remoteMode, setRemoteMode] = useState('agent');
  const activeStreamAbortRef = useRef(null);

  const loadNetInfo = useCallback(async () => {
    setIsLoadingNet(true);
    try {
      const [net, recent] = await Promise.all([
        getNetworkInfo().catch(() => null),
        getRecentSyncs(15).catch(() => ({ recent: [] })),
      ]);
      if (net) setNetworkInfo(net);
      if (recent?.recent) setRecentSyncs(recent.recent);
    } finally {
      setIsLoadingNet(false);
    }
  }, []);

  const fetchSyncItem = useCallback(async (pin) => {
    const cleanPin = String(pin || '').trim();
    if (!cleanPin) {
      setSyncError('Please enter a 4-digit Sync PIN.');
      return;
    }
    setSyncError('');
    setIsSyncLoading(true);
    activeStreamAbortRef.current?.abort();
    const controller = new AbortController();
    activeStreamAbortRef.current = controller;

    try {
      const item = await getSyncContent(cleanPin);
      setActiveContent(item);
      setIsSyncLoading(false);

      if (item.status === 'streaming') {
        void streamSyncContent(cleanPin, {
          signal: controller.signal,
          onEvent: (event) => {
            if (event.type === 'snapshot') {
              setActiveContent(event);
            } else if (event.type === 'stage') {
              setActiveContent((prev) => prev ? { ...prev, stages: [...(prev.stages || []), event] } : prev);
            } else if (event.type === 'token') {
              setActiveContent((prev) => prev ? { ...prev, content: `${prev.content || ''}${event.content || ''}` } : prev);
            } else if (event.type === 'done') {
              setActiveContent((prev) => prev ? { ...prev, status: 'completed', artifact: event.artifact || prev.artifact, model_used: event.model_used || prev.model_used, time_seconds: event.time_seconds || prev.time_seconds } : prev);
            } else if (event.type === 'error') {
              setActiveContent((prev) => prev ? { ...prev, status: 'error', error: event.detail || 'Streaming failed' } : prev);
            }
          },
        }).catch(() => {});
      }
    } catch (err) {
      setSyncError(err.message || `Sync PIN '${cleanPin}' was not found on server.`);
      setIsSyncLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      void loadNetInfo();
      if (initialSyncId) {
        setSyncIdInput(initialSyncId);
        setActiveTab('receiver');
        void fetchSyncItem(initialSyncId);
      }
    }
  }, [isOpen, initialSyncId, loadNetInfo, fetchSyncItem]);

  useEffect(() => {
    return () => {
      activeStreamAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!isOpen || !autoFollow) return;
    const timer = window.setInterval(async () => {
      try {
        const latest = await getLatestSync();
        if (latest?.sync_id && (!activeContent || activeContent.sync_id !== latest.sync_id)) {
          setSyncIdInput(latest.sync_id);
          void fetchSyncItem(latest.sync_id);
        }
      } catch {}
    }, 2500);
    return () => window.clearInterval(timer);
  }, [isOpen, autoFollow, activeContent, fetchSyncItem]);

  const copyToClipboard = async (text, key) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(''), 2000);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(''), 2000);
    }
  };

  const handleRemoteSubmit = async (e) => {
    e?.preventDefault();
    if (!remotePrompt.trim() || isSubmittingRemote) return;
    setIsSubmittingRemote(true);
    setSyncError('');
    try {
      const res = await submitRemoteTask(remotePrompt.trim(), { mode: remoteMode });
      setRemotePrompt('');
      if (res.sync_id) {
        setSyncIdInput(res.sync_id);
        void fetchSyncItem(res.sync_id);
      }
    } catch (err) {
      setSyncError(`Failed to submit task: ${err.message}`);
    } finally {
      setIsSubmittingRemote(false);
    }
  };

  if (!isOpen) return null;

  const primaryUrl = networkInfo?.recommended_frontend_url || (typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:5173');
  const isDirectHotspot = networkInfo?.is_hotspot || primaryUrl.includes('192.168.137.');

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/80 px-4 py-6 backdrop-blur-md" role="dialog" aria-modal="true" aria-label="Device Sync Hub">
      <div className="relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-3xl border border-[#ff6700]/30 bg-[#120a06] shadow-2xl shadow-[#ff6700]/10">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 bg-black/40 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#ff6700]/20 text-[#ff812b] shadow-inner">
              <Radio className="h-5 w-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-white">Hotspot & Multi-Device Hub</h2>
                <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[10px] font-mono font-medium text-emerald-300">Port Forwarding Active</span>
              </div>
              <p className="text-xs text-gray-400">Server runs on main PC • Secondary devices enter PIN to get content</p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-full p-2 text-gray-400 transition-colors hover:bg-white/10 hover:text-white" title="Close"><X className="h-5 w-5" /></button>
        </div>

        {/* Tab Selector */}
        <div className="flex border-b border-white/10 bg-black/20 px-6 pt-3">
          <button
            type="button"
            onClick={() => setActiveTab('host')}
            className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${activeTab === 'host' ? 'border-[#ff6700] text-[#ff812b]' : 'border-transparent text-gray-400 hover:text-gray-200'}`}
          >
            <Laptop className="h-4 w-4" />
            <span>Host Showcase (Main PC)</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('receiver')}
            className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${activeTab === 'receiver' ? 'border-[#ff6700] text-[#ff812b]' : 'border-transparent text-gray-400 hover:text-gray-200'}`}
          >
            <Smartphone className="h-4 w-4" />
            <span>Receiver Device (Enter PIN & View)</span>
            {activeContent && <span className="flex h-2 w-2 rounded-full bg-[#ff6700] animate-ping" />}
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeTab === 'host' ? (
            <div className="space-y-6">
              {/* Host URL Card */}
              <div className="rounded-2xl border border-[#ff6700]/25 bg-gradient-to-r from-[#211007] to-black p-5 shadow-lg">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[#ff812b]">
                      <Wifi className="h-3.5 w-3.5" />
                      {isDirectHotspot ? 'Windows Mobile Hotspot Active' : 'Local Wi-Fi / Hotspot Access URL'}
                    </div>
                    <p className="mt-1 text-lg font-mono font-bold text-white break-all">{primaryUrl}</p>
                    <p className="mt-1 text-xs text-gray-400">Open this address on phones, tablets, or laptops connected to this hotspot.</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={() => copyToClipboard(primaryUrl, 'primaryUrl')}
                      className="flex items-center gap-2 rounded-xl bg-[#ff6700] px-4 py-2.5 text-xs font-semibold text-[#1f0d03] shadow-md transition-all hover:bg-[#ff812b] hover:scale-105 active:scale-95"
                    >
                      {copiedKey === 'primaryUrl' ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                      <span>{copiedKey === 'primaryUrl' ? 'Copied URL!' : 'Copy Hotspot Link'}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => loadNetInfo()}
                      className="rounded-xl border border-white/10 bg-white/5 p-2.5 text-gray-400 hover:text-white"
                      title="Refresh network interfaces"
                    >
                      <RefreshCw className={`h-4 w-4 ${isLoadingNet ? 'animate-spin' : ''}`} />
                    </button>
                  </div>
                </div>

                {networkInfo?.all_ips && networkInfo.all_ips.length > 1 && (
                  <div className="mt-4 border-t border-white/5 pt-3">
                    <p className="text-[11px] text-gray-500 mb-1.5">Alternative Host IP Addresses:</p>
                    <div className="flex flex-wrap gap-2">
                      {networkInfo.all_ips.map((ip) => {
                        const url = `http://${ip}:5173`;
                        return (
                          <button
                            key={ip}
                            type="button"
                            onClick={() => copyToClipboard(url, `ip-${ip}`)}
                            className="inline-flex items-center gap-1.5 rounded-lg border border-white/5 bg-black/40 px-2.5 py-1 text-xs font-mono text-gray-300 hover:border-[#ff6700]/30 hover:text-[#ff812b]"
                          >
                            <span>{ip}</span>
                            <span className="text-[10px] text-gray-500">{copiedKey === `ip-${ip}` ? '✓' : 'copy'}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* Active Broadcast PIN */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-1 rounded-2xl border border-white/10 bg-black/40 p-4 flex flex-col justify-between">
                  <div>
                    <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Latest Live PIN</span>
                    <div className="mt-2 text-3xl font-mono font-extrabold text-[#ff6700] tracking-wider">
                      #{recentSyncs[0]?.sync_id || '----'}
                    </div>
                    <p className="mt-1 text-xs text-gray-500">Receiver devices can enter this 4-digit PIN to load the current response.</p>
                  </div>
                  {recentSyncs[0]?.sync_id && (
                    <button
                      type="button"
                      onClick={() => {
                        const link = `${primaryUrl}?sync=${recentSyncs[0].sync_id}`;
                        copyToClipboard(link, 'directLink');
                      }}
                      className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-white/5 px-3 py-2 text-xs font-medium text-orange-200 hover:bg-[#ff6700]/20"
                    >
                      <Share2 className="h-3.5 w-3.5" />
                      <span>{copiedKey === 'directLink' ? 'Copied Direct Link!' : 'Copy Direct Link'}</span>
                    </button>
                  )}
                </div>

                <div className="md:col-span-2 rounded-2xl border border-white/10 bg-black/40 p-4">
                  <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">How Devices & Judges Connect</span>
                  <div className="mt-3 space-y-2.5 text-xs text-gray-300">
                    <div className="flex items-start gap-2.5">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#ff6700]/20 font-mono text-[10px] font-bold text-[#ff812b]">1</span>
                      <span>Turn on <strong>Mobile Hotspot</strong> on this PC and connect receiver phones/laptops to the Wi-Fi.</span>
                    </div>
                    <div className="flex items-start gap-2.5">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#ff6700]/20 font-mono text-[10px] font-bold text-[#ff812b]">2</span>
                      <span>On the receiver device, open browser and navigate to <code className="rounded bg-black/60 px-1.5 py-0.5 font-mono text-[#ff812b]">{primaryUrl}</code></span>
                    </div>
                    <div className="flex items-start gap-2.5">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#ff6700]/20 font-mono text-[10px] font-bold text-[#ff812b]">3</span>
                      <span>Enter the <strong>4-digit Sync PIN</strong> (or tap <strong>Auto-Follow</strong>) to see live stream, generated files & answers!</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recent Broadcasts List */}
              <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400">Recent Sync Broadcasts</h3>
                  <span className="text-xs text-gray-500">{recentSyncs.length} items</span>
                </div>
                {recentSyncs.length === 0 ? (
                  <p className="text-center py-6 text-sm text-gray-500">No questions asked yet. Ask a question to generate a shareable Sync PIN!</p>
                ) : (
                  <div className="divide-y divide-white/5 max-h-48 overflow-y-auto">
                    {recentSyncs.map((item) => (
                      <div key={item.sync_id} className="flex items-center justify-between py-2.5 hover:bg-white/[0.02] px-2 rounded-lg">
                        <div className="min-w-0 flex-1 pr-3">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold text-[#ff6700] text-xs">#{item.sync_id}</span>
                            <span className="rounded bg-white/5 px-1.5 py-0.2 text-[10px] text-gray-400 uppercase">{item.category}</span>
                            {item.artifact && <span className="rounded bg-emerald-500/10 px-1.5 py-0.2 text-[10px] text-emerald-400 font-medium">📎 {item.artifact.filename}</span>}
                          </div>
                          <p className="text-xs text-gray-300 truncate mt-0.5">{item.query}</p>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          <button
                            type="button"
                            onClick={() => {
                              setActiveTab('receiver');
                              setSyncIdInput(item.sync_id);
                              void fetchSyncItem(item.sync_id);
                            }}
                            className="rounded-lg bg-[#ff6700]/10 px-2.5 py-1 text-xs font-medium text-[#ff812b] hover:bg-[#ff6700]/20"
                          >
                            View Content
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Receiver Mode */
            <div className="space-y-6">
              {/* PIN Input & Auto-Follow Header */}
              <div className="rounded-2xl border border-[#ff6700]/20 bg-black/50 p-5 shadow-lg">
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                  <div className="relative flex-1">
                    <input
                      type="text"
                      value={syncIdInput}
                      onChange={(e) => setSyncIdInput(e.target.value.replace(/[^0-9a-zA-Z_-]/g, ''))}
                      onKeyDown={(e) => { if (e.key === 'Enter') void fetchSyncItem(syncIdInput); }}
                      placeholder="Enter 4-Digit Sync PIN (e.g. 4821)"
                      className="w-full rounded-xl border border-white/10 bg-black/60 px-4 py-3 text-base font-mono font-medium text-white placeholder-gray-500 outline-none focus:border-[#ff6700]"
                      maxLength={12}
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => void fetchSyncItem(syncIdInput)}
                    disabled={!syncIdInput.trim() || isSyncLoading}
                    className="flex items-center justify-center gap-2 rounded-xl bg-[#ff6700] px-6 py-3 text-sm font-semibold text-[#1a0c04] shadow-md transition-all hover:bg-[#ff812b] disabled:opacity-40"
                  >
                    {isSyncLoading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                    <span>Get Content</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setAutoFollow((prev) => !prev)}
                    className={`flex items-center justify-center gap-2 rounded-xl border px-4 py-3 text-xs font-semibold transition-all ${autoFollow ? 'border-red-500 bg-red-500/20 text-red-300' : 'border-white/10 bg-white/5 text-gray-300 hover:bg-white/10'}`}
                    title="Automatically display whatever query the main presenter runs on the server"
                  >
                    <Radio className={`h-4 w-4 ${autoFollow ? 'animate-ping text-red-400' : 'text-gray-400'}`} />
                    <span>{autoFollow ? 'Auto-Follow ON' : 'Auto-Follow'}</span>
                  </button>
                </div>

                {syncError && (
                  <div className="mt-3 flex items-center gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl p-2.5">
                    <AlertTriangle className="h-4 w-4 shrink-0" />
                    <span>{syncError}</span>
                  </div>
                )}
              </div>

              {/* Remote Task Submitter Drawer */}
              <div className="rounded-2xl border border-white/10 bg-[#170e08]/60 p-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-orange-200 flex items-center gap-2 mb-2">
                  <Send className="h-3.5 w-3.5 text-[#ff6700]" />
                  Ask Question from this Device (Processes on Main Server GPU)
                </span>
                <form onSubmit={handleRemoteSubmit} className="flex gap-2">
                  <input
                    type="text"
                    value={remotePrompt}
                    onChange={(e) => setRemotePrompt(e.target.value)}
                    placeholder="Type task or prompt to execute on host server..."
                    className="flex-1 rounded-xl border border-white/10 bg-black/50 px-3 py-2 text-xs text-white placeholder-gray-500 outline-none focus:border-[#ff6700]"
                  />
                  <button
                    type="submit"
                    disabled={!remotePrompt.trim() || isSubmittingRemote}
                    className="flex items-center gap-1.5 rounded-xl bg-[#ff6700]/20 px-3 py-2 text-xs font-semibold text-[#ff812b] hover:bg-[#ff6700]/30 disabled:opacity-30"
                  >
                    {isSubmittingRemote ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
                    <span>Submit</span>
                  </button>
                </form>
              </div>

              {/* Displayed Content */}
              {activeContent ? (
                <div className="rounded-2xl border border-white/[0.08] bg-[#140d09] p-5 shadow-xl space-y-4">
                  {/* Status Banner */}
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-bold text-[#ff6700]">PIN #{activeContent.sync_id}</span>
                      <span className="rounded-full bg-[#ff6700]/10 px-2 py-0.5 text-[11px] font-medium text-orange-200 uppercase">{activeContent.category || 'general'}</span>
                      {activeContent.model_used && <span className="rounded-full bg-white/5 px-2 py-0.5 text-[11px] text-gray-400">{activeContent.model_used}</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      {activeContent.status === 'streaming' ? (
                        <span className="flex items-center gap-1.5 text-xs font-mono text-[#ff812b]">
                          <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                          Live Streaming...
                        </span>
                      ) : activeContent.status === 'error' ? (
                        <span className="flex items-center gap-1.5 text-xs font-mono text-red-400">
                          <AlertTriangle className="h-3.5 w-3.5" />
                          Failed
                        </span>
                      ) : (
                        <span className="flex items-center gap-1.5 text-xs font-mono text-emerald-400">
                          <Check className="h-3.5 w-3.5" />
                          Completed ({activeContent.time_seconds || 0}s)
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Query */}
                  <div>
                    <span className="text-[11px] font-medium uppercase tracking-wider text-gray-500">Query / Request:</span>
                    <p className="mt-1 font-medium text-sm text-gray-100 bg-black/40 rounded-xl p-3 border border-white/5">{activeContent.query}</p>
                  </div>

                  {/* Stages Timeline */}
                  {activeContent.stages && activeContent.stages.length > 0 && (
                    <div className="rounded-xl border border-[#ff6700]/15 bg-black/30 p-3">
                      <div className="flex items-center gap-2 text-xs font-medium text-orange-200">
                        {activeContent.status === 'streaming' ? <LoaderCircle className="h-3.5 w-3.5 animate-spin text-[#ff6700]" /> : <Check className="h-3.5 w-3.5 text-emerald-400" />}
                        <span>{describeStage(activeContent.stages.at(-1)).title}: {describeStage(activeContent.stages.at(-1)).detail}</span>
                      </div>
                    </div>
                  )}

                  {/* Formatted Content */}
                  {activeContent.content && (
                    <div className="space-y-3 pt-2">
                      <FormattedAnswer content={activeContent.content} copiedId={copiedKey} onCopied={(id) => setCopiedKey(id)} messageId={activeContent.sync_id} />
                    </div>
                  )}

                  {/* Download Artifact */}
                  {activeContent.artifact?.download_url && (
                    <div className="pt-3 border-t border-white/5">
                      <a
                        href={backendUrl(activeContent.artifact.download_url)}
                        download
                        className="inline-flex items-center gap-2 rounded-xl bg-[#ff6700] px-4 py-2.5 text-xs font-bold text-[#1f0d03] shadow-lg transition-all hover:bg-[#ff812b] hover:scale-105"
                      >
                        <Download className="h-4 w-4" />
                        <span>Download {activeContent.artifact.filename || 'Generated File'}</span>
                      </a>
                    </div>
                  )}

                  {/* Sources */}
                  {activeContent.sources && activeContent.sources.length > 0 && (
                    <div className="pt-3 border-t border-white/5">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-400 mb-2">
                        <Database className="h-3.5 w-3.5" />
                        <span>Knowledge Base Sources</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {activeContent.sources.map((src, idx) => (
                          <span key={idx} className="rounded bg-white/5 px-2 py-0.5 text-xs text-gray-300">
                            {typeof src === 'string' ? src : src.filename || src.source || `Source ${idx+1}`}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12 rounded-2xl border border-dashed border-white/10 bg-black/20 p-6">
                  <Smartphone className="h-10 w-10 text-gray-600 mx-auto mb-3 animate-bounce" />
                  <p className="text-sm font-medium text-gray-300">Enter a 4-Digit Sync PIN above</p>
                  <p className="text-xs text-gray-500 mt-1">Or toggle Auto-Follow to automatically stream the newest answer from the host server.</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-white/10 bg-black/40 px-6 py-3 flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-2">
            <Radio className="h-3.5 w-3.5 text-emerald-400" />
            <span>Agent OTG Air-Gapped Network Sync</span>
          </div>
          <button type="button" onClick={onClose} className="rounded-xl bg-white/10 px-4 py-1.5 text-xs font-medium text-white hover:bg-white/20">Close</button>
        </div>

      </div>
    </div>
  );
}

export default function AgentOTG() {
  const [autoNewChatOnStart, setAutoNewChatOnStart] = useState(() => readStoredJson(AUTO_NEW_CHAT_STORAGE_KEY, true));
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDeviceHubOpen, setIsDeviceHubOpen] = useState(false);
  const [selectedSyncId, setSelectedSyncId] = useState('');
  const [latestSyncId, setLatestSyncId] = useState('');
  const [chats, setChats] = useState(() => {
    const shouldStartNew = readStoredJson(AUTO_NEW_CHAT_STORAGE_KEY, true);
    if (shouldStartNew) return [createChat()];
    const saved = readStoredJson(CHAT_STORAGE_KEY, []);
    return Array.isArray(saved) && saved.length ? saved : [createChat()];
  });
  const [activeChatId, setActiveChatId] = useState(() => {
    const shouldStartNew = readStoredJson(AUTO_NEW_CHAT_STORAGE_KEY, true);
    if (shouldStartNew) return '';
    return window.localStorage.getItem(ACTIVE_CHAT_STORAGE_KEY) || '';
  });
  const [serverSessions, setServerSessions] = useState([]);

  useEffect(() => {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const syncParam = urlParams.get('sync') || urlParams.get('id');
      if (syncParam) {
        setSelectedSyncId(syncParam);
        setIsDeviceHubOpen(true);
      }
    } catch {}
  }, []);

  const [titleOverrides, setTitleOverrides] = useState(() => readStoredJson(TITLE_STORAGE_KEY, {}));
  const [hiddenSessionIds, setHiddenSessionIds] = useState(() => readStoredJson(HIDDEN_SESSIONS_STORAGE_KEY, []));
  const [sessionTitleCache, setSessionTitleCache] = useState(() => readStoredJson(SESSION_TITLE_CACHE_KEY, {}));
  const [pinnedHistoryIds, setPinnedHistoryIds] = useState(() => readStoredJson(PINNED_HISTORY_KEY, []));
  const [ragMemory, setRagMemory] = useState(() => readStoredJson(RAG_MEMORY_KEY, []));
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [activeFilter, setActiveFilter] = useState(null);
  const [mode, setMode] = useState('agent');
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => window.innerWidth >= 768);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isModeMenuOpen, setIsModeMenuOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [visibleHistoryCount, setVisibleHistoryCount] = useState(HISTORY_PAGE_SIZE);
  const [editingHistoryId, setEditingHistoryId] = useState(null);
  const [editedTitle, setEditedTitle] = useState('');
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [attachmentNotice, setAttachmentNotice] = useState('');
  const [historyError, setHistoryError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');
  const [typedHeadingCount, setTypedHeadingCount] = useState(0);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [copiedId, setCopiedId] = useState(null);

  const fileInputRef = useRef(null);
  const menuRef = useRef(null);
  const modeMenuRef = useRef(null);
  const textareaRef = useRef(null);
  const searchInputRef = useRef(null);
  const abortControllerRef = useRef(null);
  const messageEndRef = useRef(null);
  const scrollContainerRef = useRef(null);
  const hydratingSessionIdsRef = useRef(new Set());
  const activeChat = chats.find((chat) => chat.id === activeChatId) || chats[0];

  const updateChat = useCallback((chatId, updater) => {
    setChats((currentChats) => currentChats.map((chat) => (chat.id === chatId ? updater(chat) : chat)));
  }, []);

  const loadServerSessions = useCallback(async () => {
    try {
      const payload = await getSessions();
      // Do not fetch every session transcript here. A full fetch for all history
      // entries caused a burst of requests on every reload and overloaded FastAPI.
      setServerSessions((payload.sessions || []).filter((session) => Number(session.message_count || 0) > 0));
      setHistoryError('');
    } catch (error) {
      setHistoryError(error.message || 'The saved chats could not be loaded.');
    }
  }, []);

 const checkBackend = useCallback(async () => {
  if (!navigator.onLine) {
    setApiStatus('offline');
    return;
  }
  try {
    const health = await getHealth();
    setApiStatus(health.ollama === 'connected' ? 'online' : 'degraded');
  } catch {
    setApiStatus('offline');
  }
}, []);
useEffect(() => {
  function handleOnline() { setIsOnline(true); void checkBackend(); }
  function handleOffline() { setIsOnline(false); setApiStatus('offline'); }
  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);
  return () => {
    window.removeEventListener('online', handleOnline);
    window.removeEventListener('offline', handleOffline);
  };
}, [checkBackend]);
  const hydrateSessionTitles = useCallback(async (sessions) => {
    const missing = sessions.filter((session) => !titleOverrides[session.session_name] && !sessionTitleCache[session.session_name] && !hydratingSessionIdsRef.current.has(session.session_name));
    for (const session of missing) {
      hydratingSessionIdsRef.current.add(session.session_name);
      try {
        const detail = await getSessionMessages(session.session_name);
        const firstPrompt = (detail.messages || [])
          .find((message) => message.role === 'user' && message.content?.trim())?.content;
        if (!firstPrompt) {
          setHiddenSessionIds((current) => current.includes(session.session_name) ? current : [...current, session.session_name]);
        } else {
          setSessionTitleCache((current) => ({ ...current, [session.session_name]: deriveTitle(firstPrompt) }));
        }
      } catch {
        // Keep a stable fallback so a temporarily unavailable API is not retried
        // for every render. A manual history refresh can retry later.
        setSessionTitleCache((current) => ({ ...current, [session.session_name]: 'Saved conversation' }));
      } finally {
        hydratingSessionIdsRef.current.delete(session.session_name);
      }
    }
  }, [sessionTitleCache, titleOverrides]);

  const handleNewChat = useCallback(() => {
    if (isProcessing) return;

    setAttachments([]);
    setRagMemory([]);
    writeStoredJson(RAG_MEMORY_KEY, []);
    setInput('');
    setActiveFilter(null);
    setAttachmentNotice('');

    setChats((currentChats) => {
      const active = currentChats.find((chat) => chat.id === activeChatId);
      if (active && active.messages.length === 0) {
        return currentChats;
      }
      const newChat = createChat();
      setActiveChatId(newChat.id);
      return [...currentChats, newChat];
    });

    textareaRef.current?.focus();

    resetConversation()
      .then(() => {
        setApiStatus('online');
        void loadServerSessions();
      })
      .catch(() => {
        // Local chat creation still works if backend is restarting.
      });
  }, [activeChatId, isProcessing, loadServerSessions]);

  useEffect(() => { writeStoredJson(CHAT_STORAGE_KEY, chats); }, [chats]);
  useEffect(() => { if (activeChat?.id) window.localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, activeChat.id); }, [activeChat?.id]);
  useEffect(() => { writeStoredJson(TITLE_STORAGE_KEY, titleOverrides); }, [titleOverrides]);
  useEffect(() => { writeStoredJson(HIDDEN_SESSIONS_STORAGE_KEY, hiddenSessionIds); }, [hiddenSessionIds]);
  useEffect(() => { writeStoredJson(SESSION_TITLE_CACHE_KEY, sessionTitleCache); }, [sessionTitleCache]);
  useEffect(() => { writeStoredJson(PINNED_HISTORY_KEY, pinnedHistoryIds); }, [pinnedHistoryIds]);
  useEffect(() => { writeStoredJson(RAG_MEMORY_KEY, ragMemory); }, [ragMemory]);
  useEffect(() => { writeStoredJson(AUTO_NEW_CHAT_STORAGE_KEY, autoNewChatOnStart); }, [autoNewChatOnStart]);


  useEffect(() => {
    const initialize = window.setTimeout(() => {
      void loadServerSessions();
      void checkBackend();
    }, 0);
    const timer = window.setInterval(checkBackend, 30_000);
    return () => {
      window.clearTimeout(initialize);
      window.clearInterval(timer);
    };
  }, [checkBackend, loadServerSessions]);

  useEffect(() => {
    const candidates = serverSessions
      .filter((session) => !hiddenSessionIds.includes(session.session_name))
      .sort((a, b) => new Date(b.last_activity || b.created_at) - new Date(a.last_activity || a.created_at))
      .slice(0, visibleHistoryCount);
    const timer = window.setTimeout(() => { void hydrateSessionTitles(candidates); }, 0);
    return () => window.clearTimeout(timer);
  }, [hiddenSessionIds, hydrateSessionTitles, serverSessions, visibleHistoryCount]);

  useEffect(() => {
    const clickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) setIsMenuOpen(false);
      if (modeMenuRef.current && !modeMenuRef.current.contains(event.target)) setIsModeMenuOpen(false);
    };
    document.addEventListener('mousedown', clickOutside);
    return () => document.removeEventListener('mousedown', clickOutside);
  }, []);

  useEffect(() => {
    const shortcut = (event) => {
      const modifier = event.ctrlKey || event.metaKey;
      if (modifier && event.key.toLowerCase() === 'n') {
        event.preventDefault();
        event.stopPropagation();
        if (!isProcessing) handleNewChat();
      }
      if (modifier && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        event.stopPropagation();
        setIsSearchOpen(true);
      }
      if (modifier && event.key === '/') {
        event.preventDefault();
        event.stopPropagation();
        textareaRef.current?.focus();
      }
      if (modifier && event.shiftKey && event.key.toLowerCase() === 'm') {
        event.preventDefault();
        event.stopPropagation();
        selectMode(mode === 'agent' ? 'chief' : 'agent');
      }
      if (event.key === 'Escape') {
        setIsMenuOpen(false);
        setIsModeMenuOpen(false);
        setIsSearchOpen(false);
        setEditingHistoryId(null);
      }
    };
    window.addEventListener('keydown', shortcut, true);
    return () => window.removeEventListener('keydown', shortcut, true);
  }, [handleNewChat, isProcessing, mode]);

  useEffect(() => { if (isSearchOpen) window.setTimeout(() => searchInputRef.current?.focus(), 0); }, [isSearchOpen]);
  const lastMessageContent = activeChat?.messages.at(-1)?.content;
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    container.scrollTo({ top: container.scrollHeight, behavior: isProcessing ? 'auto' : 'smooth' });
  }, [activeChat?.messages.length, isProcessing, lastMessageContent]);
  useEffect(() => {
    if (!copiedId) return undefined;
    const timer = window.setTimeout(() => setCopiedId(null), 1500);
    return () => window.clearTimeout(timer);
  }, [copiedId]);
useEffect(() => {
  const noMessages = !(activeChat?.messages?.length > 0);
  if (!noMessages) return undefined;
  const fullLength = HEADING_PREFIX.length + HEADING_HIGHLIGHT.length + HEADING_SUFFIX.length;
  setTypedHeadingCount(0);
  const interval = setInterval(() => {
    setTypedHeadingCount((count) => {
      if (count >= fullLength) {
        clearInterval(interval);
        return count;
      }
      return count + 1;
    });
  }, 60);
  return () => clearInterval(interval);
}, [activeChat?.id, activeChat?.messages?.length]);
  const historyItems = useMemo(() => {
    const localItems = chats.filter((chat) => chat.messages.length > 0).map((chat) => ({ id: chat.id, type: 'local', title: chat.title || 'New chat', date: chat.updatedAt, chat, pinKey: `local-${chat.id}` }));
    const linkedSessions = new Set(chats.map((chat) => chat.backendSessionName).filter(Boolean));
    const serverItems = serverSessions
      .filter((session) => !hiddenSessionIds.includes(session.session_name) && !linkedSessions.has(session.session_name))
      .map((session) => ({
        id: `server-${session.session_name}`,
        type: 'server',
        title: titleOverrides[session.session_name] || sessionTitleCache[session.session_name] || 'Loading title…',
        date: session.last_activity || session.created_at,
        session,
        pinKey: `server-${session.session_name}`,
      }));
    return [...localItems, ...serverItems]
      .map((item) => ({ ...item, pinned: pinnedHistoryIds.includes(item.pinKey) }))
      .sort((a, b) => Number(b.pinned) - Number(a.pinned) || new Date(b.date) - new Date(a.date));
  }, [chats, hiddenSessionIds, pinnedHistoryIds, serverSessions, sessionTitleCache, titleOverrides]);

  const matchingHistory = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    return term ? historyItems.filter((item) => item.title.toLowerCase().includes(term)) : historyItems;
  }, [historyItems, searchTerm]);

  function appendStage(chatId, messageId, stage) {
    updateChat(chatId, (chat) => ({
      ...chat,
      updatedAt: new Date().toISOString(),
      messages: chat.messages.map((message) => {
        if (message.id !== messageId) return message;
        const stages = [...(message.stages || [])];
        if (stages.at(-1)?.detail !== stage.detail) stages.push(stage);
        return { ...message, stages };
      }),
    }));
  }

  function patchAssistantMessage(chatId, messageId, patch) {
    updateChat(chatId, (chat) => ({
      ...chat,
      updatedAt: new Date().toISOString(),
      messages: chat.messages.map((message) => message.id === messageId ? { ...message, ...patch } : message),
    }));
  }

  async function associateBackendSession(chatId) {
    try {
      const payload = await getSessions();
      const session = payload.sessions?.[0];
      if (session?.session_name) updateChat(chatId, (chat) => ({ ...chat, backendSessionName: session.session_name }));
      await loadServerSessions();
    } catch {
      // The local chat still contains the completed answer if the session refresh fails.
    }
  }

  async function tryIngestAttachments(chatId, messageId, selectedAttachments, signal) {
    if (!selectedAttachments.length) return;
    appendStage(chatId, messageId, { label: 'attachments', detail: `Indexing ${selectedAttachments.length} file${selectedAttachments.length === 1 ? '' : 's'} in knowledge base…` });

    // Attempt backend file upload (works for browser File objects)
    try {
      const res = await uploadKnowledgeBase(selectedAttachments, { signal });
      const totalIndexed = res.files_indexed || selectedAttachments.length;
      setRagMemory((current) => {
        const additions = selectedAttachments.map((file) => ({
          key: `${file.name}-${file.size}-${file.lastModified}`,
          name: file.name,
          text: '',
          indexed: true,
          updatedAt: new Date().toISOString(),
        }));
        const retained = current.filter((entry) => !additions.some((next) => next.key === entry.key));
        return [...retained, ...additions].slice(-24);
      });
      appendStage(chatId, messageId, { label: 'attachments', detail: `Indexed ${totalIndexed} file${totalIndexed === 1 ? '' : 's'} into ChromaDB knowledge base.` });
      setAttachmentNotice('');
      return;
    } catch (uploadError) {
      // Fallback: local file path ingestion if available (Electron)
      const localPaths = selectedAttachments.map((file) => file.path).filter((path) => typeof path === 'string' && path.trim());
      if (localPaths.length === selectedAttachments.length) {
        try {
          await ingestKnowledgeBase(localPaths);
          appendStage(chatId, messageId, { label: 'attachments', detail: 'Files indexed. The agent can now retrieve them as context.' });
          setAttachmentNotice('');
          return;
        } catch (error) {
          const warning = `Files could not be indexed: ${error.message}`;
          setAttachmentNotice(warning);
          appendStage(chatId, messageId, { label: 'attachments', detail: warning });
          return;
        }
      }

      const readableCount = selectedAttachments.filter((file) => ragMemory.some((entry) => entry.key === `${file.name}-${file.size}-${file.lastModified}`)).length;
      if (readableCount) {
        appendStage(chatId, messageId, {
          label: 'attachments',
          detail: `${readableCount} text file${readableCount === 1 ? '' : 's'} kept in browser context. PDF/Office files still need backend indexing — run start-demo.ps1 and ensure Ollama is running.`,
        });
        return;
      }
      const warning = `Could not index file(s): ${uploadError.message}. Install backend deps (python-multipart) and start Ollama with nomic-embed-text.`;
      setAttachmentNotice(warning);
      appendStage(chatId, messageId, { label: 'attachments', detail: warning });
          throw new Error(warning, { cause: uploadError });
    }
  }

  async function handleSend() {
    if (!input.trim() || isProcessing || !activeChat) return;
    if (mode === 'chief' && attachments.length) {
      setAttachmentNotice('Chief mode is text-only. Switch to Agent mode to work with files.');
      return;
    }
    const prompt = input.trim();
    const selectedAttachments = [...attachments];
    const selectedFilter = activeFilter;
    const requestPrompt = buildSubmissionPrompt(
      prompt,
      selectedFilter,
      mode,
      activeChat.messages,
      attachmentContext(selectedAttachments, ragMemory),
      selectedAttachments.length > 0 || Boolean(activeChat?.isRagEnabled),
    );
    const chatId = activeChat.id;
    const userMessage = {
      id: createId('message'), role: 'user', content: prompt, createdAt: new Date().toISOString(),
      attachments: selectedAttachments.map((file) => ({ name: file.name, type: file.type })),
      documentType: selectedFilter, mode,
    };
    const assistantMessage = {
      id: createId('message'), role: 'assistant', content: '', createdAt: new Date().toISOString(), streaming: true,
      stages: [{ label: 'understanding', detail: mode === 'chief' ? 'Complex-task routing selected; preparing the Qwen 14B request.' : 'Classifying intent, checking conversation context, and selecting the appropriate workflow.' }], meta: {},
    };

    updateChat(chatId, (chat) => {
      const firstPrompt = chat.messages.find((message) => message.role === 'user')?.content || prompt;
      return {
        ...chat,
        title: !chat.titleEdited ? deriveTitle(firstPrompt) : chat.title,
        updatedAt: new Date().toISOString(),
        messages: [...chat.messages, userMessage, assistantMessage],
      };
    });
    setInput('');
    setAttachments([]);
    setActiveFilter(null);
    setAttachmentNotice('');
    setIsProcessing(true);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      if (mode === 'agent' && selectedAttachments.length === 1 && isImageFile(selectedAttachments[0]) && !selectedFilter) {
        appendStage(chatId, assistantMessage.id, { label: 'vision', detail: 'Analyzing the attached image with Qwen VL…' });
        const image_b64 = await fileToBase64(selectedAttachments[0]);
        if (controller.signal.aborted) throw new DOMException('Generation stopped.', 'AbortError');
        const result = await askAboutImage(prompt, image_b64, { signal: controller.signal });
        patchAssistantMessage(chatId, assistantMessage.id, { content: result.answer || 'Image analysis completed.', streaming: false, meta: { modelUsed: result.model_used, timeSeconds: result.time_seconds } });
      } else {
        if (mode === 'agent') await tryIngestAttachments(chatId, assistantMessage.id, selectedAttachments, controller.signal);
        if (controller.signal.aborted) throw new DOMException('Generation stopped.', 'AbortError');
        let streamError = '';
        await streamQuestion(requestPrompt, {
          signal: controller.signal,
          onEvent: (event) => {
            if (event.type === 'sync_info') {
              patchAssistantMessage(chatId, assistantMessage.id, { syncId: event.sync_id });
              setLatestSyncId(event.sync_id);
            } else if (event.type === 'stage') {
              appendStage(chatId, assistantMessage.id, { label: event.label, detail: event.detail });
            } else if (event.type === 'plan') {
              const tasks = (event.sub_tasks || []).map((task) => task.label).join(' · ');
              appendStage(chatId, assistantMessage.id, { label: 'plan', detail: tasks ? `Plan: ${tasks}` : 'Plan created.' });
            } else if (event.type === 'token') {
              const token = String(event.content || '');
              if (isModelConnectionError(token)) {
                streamError = friendlyErrorMessage(token);
                return;
              }
              updateChat(chatId, (chat) => ({
                ...chat,
                updatedAt: new Date().toISOString(),
                messages: chat.messages.map((message) => message.id === assistantMessage.id ? { ...message, content: `${message.content || ''}${token}` } : message),
              }));
            } else if (event.type === 'done') {
              patchAssistantMessage(chatId, assistantMessage.id, {
                streaming: false,
                syncId: event.sync_id || assistantMessage.syncId,
                meta: { modelUsed: event.model_used || (mode === 'chief' ? 'qwen2.5:14b' : undefined), timeSeconds: event.time_seconds, artifact: event.artifact, sources: event.sources },
              });
            } else if (event.type === 'error') {
              streamError = event.detail || 'The backend could not complete this request.';
            }
          },
        });
        if (streamError) throw new Error(streamError);
        patchAssistantMessage(chatId, assistantMessage.id, { streaming: false });
      }
      setApiStatus('online');
      void associateBackendSession(chatId);
    } catch (error) {
      const wasCancelled = controller.signal.aborted || error.name === 'AbortError';
      const errorMessage = wasCancelled ? 'Generation stopped.' : friendlyErrorMessage(error.message || error);
      patchAssistantMessage(chatId, assistantMessage.id, { streaming: false, error: errorMessage });
      if (!wasCancelled) setApiStatus(isModelConnectionError(errorMessage) ? 'degraded' : 'offline');
    } finally {
      if (abortControllerRef.current === controller) abortControllerRef.current = null;
      setIsProcessing(false);
    }
  }

  function handleStop() {
    abortControllerRef.current?.abort();
  }

  async function openHistoryItem(item) {
    setIsSearchOpen(false);
    if (item.type === 'local') {
      setActiveChatId(item.chat.id);
      if (item.chat.backendSessionName) {
        try {
          const detail = await getSessionMessages(item.chat.backendSessionName);
          updateChat(item.chat.id, (chat) => ({ ...chat, messages: toUiMessages(detail.messages), updatedAt: new Date().toISOString() }));
        } catch {
          // Local data is retained when a stored session cannot be refreshed.
        }
      }
      return;
    }
    let messages;
    try {
      messages = (await getSessionMessages(item.session.session_name)).messages;
    } catch (error) {
      setHistoryError(friendlyErrorMessage(error.message || error));
      return;
    }
    const existing = chats.find((chat) => chat.backendSessionName === item.session.session_name);
    if (existing) {
      updateChat(existing.id, (chat) => ({ ...chat, messages: toUiMessages(messages), updatedAt: new Date().toISOString() }));
      setActiveChatId(existing.id);
      return;
    }
    const loadedChat = {
      id: createId(), title: item.title, titleEdited: Boolean(titleOverrides[item.session.session_name]), backendSessionName: item.session.session_name,
      createdAt: item.session.created_at, updatedAt: item.session.last_activity || item.session.created_at, messages: toUiMessages(messages),
    };
    setChats((currentChats) => [...currentChats, loadedChat]);
    setActiveChatId(loadedChat.id);
  }

  function selectMode(nextMode) {
    setMode(nextMode);
    setIsModeMenuOpen(false);
    if (nextMode === 'chief') {
      setActiveFilter(null);
      setAttachments([]);
      setAttachmentNotice('Chief mode uses Qwen 14B for a complex text task. Files and document tools are available in Agent mode.');
    } else {
      setAttachmentNotice('');
    }
  }

  async function handleFileChange(event) {
    const selected = Array.from(event.target.files || []);
    setAttachments((currentFiles) => {
      const unique = [...currentFiles];
      selected.forEach((file) => {
        if (!unique.some((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified)) unique.push(file);
      });
      return unique;
    });
    event.target.value = '';
    const readable = selected.filter(isReadableTextFile);
    if (!readable.length) {
      setAttachmentNotice('File selected. It will be indexed into the shared knowledge base when you send your message.');
      return;
    }
    try {
      const entries = await Promise.all(readable.map(async (file) => ({
        key: `${file.name}-${file.size}-${file.lastModified}`,
        name: file.name,
        text: (await file.text()).slice(0, 12_000),
        updatedAt: new Date().toISOString(),
      })));
      setRagMemory((current) => {
        const retained = current.filter((entry) => !entries.some((next) => next.key === entry.key));
        return [...retained, ...entries].slice(-12);
      });
      setAttachmentNotice(`${entries.length} text file${entries.length === 1 ? '' : 's'} added to this browser's RAG memory. Ask a question now or refer to the uploaded material later in this chat.`);
    } catch {
      setAttachmentNotice('The selected text file could not be read. Try a smaller UTF-8 text, CSV, JSON, Markdown, or code file.');
    }
  }

  function handleInput(event) {
    setInput(event.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }

  function saveHistoryTitle(item) {
    const title = editedTitle.trim() || 'Untitled chat';
    if (item.type === 'local') {
      updateChat(item.chat.id, (chat) => ({ ...chat, title, titleEdited: true }));
    } else {
      setTitleOverrides((current) => ({ ...current, [item.session.session_name]: title }));
      setSessionTitleCache((current) => ({ ...current, [item.session.session_name]: title }));
    }
    setEditingHistoryId(null);
  }

  function togglePin(item) {
    setPinnedHistoryIds((current) => current.includes(item.pinKey)
      ? current.filter((key) => key !== item.pinKey)
      : [...current, item.pinKey]);
  }

  function deleteHistoryItem(item) {
    if (item.type === 'server') {
      setHiddenSessionIds((current) => current.includes(item.session.session_name) ? current : [...current, item.session.session_name]);
      setPinnedHistoryIds((current) => current.filter((key) => key !== item.pinKey));
      return;
    }
    setChats((currentChats) => {
      const remaining = currentChats.filter((chat) => chat.id !== item.chat.id);
      const nextChats = remaining.length ? remaining : [createChat()];
      if (activeChatId === item.chat.id) setActiveChatId(nextChats[0].id);
      return nextChats;
    });
    setPinnedHistoryIds((current) => current.filter((key) => key !== item.pinKey));
  }

  async function clearVisibleHistory() {
    try {
      await resetConversation();
      setApiStatus('online');
    } catch {
      // Clear the frontend list even if the API is unavailable.
    }
    setAttachments([]);
    setRagMemory([]);
    writeStoredJson(RAG_MEMORY_KEY, []);
    setAttachmentNotice('');
    setHiddenSessionIds(serverSessions.map((session) => session.session_name));
    const newChat = createChat();
    setChats([newChat]);
    setActiveChatId(newChat.id);
    setShowClearConfirm(false);
  }

  const hasMessages = activeChat?.messages.length > 0;
  const visibleHistory = historyItems.slice(0, visibleHistoryCount);

  return (
    
    <div className="flex h-screen overflow-hidden bg-[#0a0502] font-sans text-gray-100">
      <style>{`
  * {
    scrollbar-width: none;
    -ms-overflow-style: none;
  }
  *::-webkit-scrollbar {
    display: none;
  }
`}</style>
      {isSidebarOpen && <button type="button" aria-label="Close sidebar" onClick={() => setIsSidebarOpen(false)} className="fixed inset-0 z-30 bg-black/60 md:hidden" />}
      <aside className={`${isSidebarOpen ? 'w-[280px]' : 'w-0'} fixed inset-y-0 left-0 z-40 flex h-full shrink-0 flex-col overflow-hidden border-r border-white/[0.04] bg-black shadow-2xl transition-all duration-300 md:relative md:z-auto md:shadow-none`}>
        <div className="flex min-w-[280px] items-center justify-between p-4">
          <div className="flex items-center gap-2 text-[22px] text-gray-200"><img src="/logo.png" alt="Agent OTG" className="h-7 w-7 object-contain" /><span>Agent OTG</span></div>
          <button type="button" onClick={() => setIsSidebarOpen(false)} className="rounded-full p-2 text-gray-400 transition-colors hover:bg-white/10" title="Close sidebar"><ChevronsLeft className="h-5 w-5" /></button>
        </div>

        <div className="mt-2 flex min-w-[280px] flex-col gap-2 px-3">
          <button type="button" onClick={handleNewChat} disabled={isProcessing} className="flex w-full items-center gap-3 rounded-xl bg-[#ff6700]/10 px-4 py-3 text-sm font-medium text-[#ff812b] transition-colors hover:bg-[#ff6700]/20 disabled:cursor-not-allowed disabled:opacity-50"><Plus className="h-5 w-5" />New chat <span className="ml-auto text-[10px] text-orange-200/60">Ctrl N</span></button>
          <button type="button" onClick={() => setIsSearchOpen(true)} className="flex w-full items-center gap-3 rounded-xl bg-[#21120b] px-4 py-3 text-sm font-medium text-gray-300 transition-colors hover:bg-[#2e190f]"><Search className="h-5 w-5 text-gray-400" />Search chat <span className="ml-auto text-[10px] text-gray-500">Ctrl K</span></button>
          <button type="button" onClick={() => setIsDeviceHubOpen(true)} className="flex w-full items-center gap-3 rounded-xl bg-[#21120b] px-4 py-3 text-sm font-medium text-gray-300 transition-colors hover:bg-[#2e190f] hover:text-[#ff812b]"><Radio className="h-5 w-5 text-[#ff6700]" />Device Hub / Sync <span className="ml-auto flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" /></button>
          <button type="button" onClick={() => setIsSettingsOpen(true)} className="flex w-full items-center gap-3 rounded-xl bg-[#21120b] px-4 py-3 text-sm font-medium text-gray-300 transition-colors hover:bg-[#2e190f]"><Settings className="h-5 w-5 text-gray-400" />Settings</button>
          <button type="button" onClick={() => setShowClearConfirm(true)} disabled={isProcessing} className="group flex w-full items-center gap-3 rounded-xl bg-[#21120b] px-4 py-3 text-sm font-medium text-gray-300 transition-colors hover:bg-red-500/10 hover:text-red-300 disabled:cursor-not-allowed disabled:opacity-50"><Trash2 className="h-5 w-5 text-gray-400 transition-colors group-hover:text-red-400" />Clear history</button>

        </div>

        <div className="mt-6 flex-1 overflow-y-auto px-3 pb-4">
          <div className="mb-3 flex items-center justify-between px-3"><p className="text-xs font-semibold uppercase tracking-wider text-gray-500">History</p><button type="button" onClick={() => void loadServerSessions()} className="rounded p-1 text-gray-500 hover:bg-white/5 hover:text-gray-200" title="Refresh saved chats"><RotateCw className="h-3.5 w-3.5" /></button></div>
          {historyError && <p className="mb-3 px-3 text-xs leading-5 text-amber-300">Saved history is unavailable until the backend reconnects.</p>}
          <div className="space-y-0.5">
            {visibleHistory.map((item) => {
              const active = item.type === 'local' && item.chat.id === activeChat?.id;
              return (
                <div key={item.id} className={`group relative flex items-center rounded-xl ${active ? 'border border-[#ff6700]/10 bg-[#ff6700]/5' : 'hover:bg-[#21120b]'}`}>
                  {editingHistoryId === item.id ? (
                    <div className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2.5"><MessageSquare className="h-4 w-4 shrink-0 text-[#ff6700]" /><input autoFocus value={editedTitle} onChange={(event) => setEditedTitle(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') saveHistoryTitle(item); if (event.key === 'Escape') setEditingHistoryId(null); }} className="min-w-0 flex-1 bg-transparent text-sm text-white outline-none" /><button type="button" onClick={() => saveHistoryTitle(item)} className="rounded p-1 text-emerald-400 hover:bg-emerald-400/10" title="Save title"><Check className="h-3.5 w-3.5" /></button></div>
                  ) : (
                    <>
                      <button type="button" onClick={() => void openHistoryItem(item)} className="flex min-w-0 flex-1 items-center gap-3 px-3 py-2.5 text-left text-sm text-gray-300">
                        <MessageSquare className={`h-4 w-4 shrink-0 ${active ? 'text-[#ff6700]' : 'text-gray-500'}`} />
                        <span className="min-w-0 flex-1 truncate">{item.title}</span>
                        {item.pinned && <Pin className="h-3 w-3 shrink-0 fill-[#ff6700] text-[#ff6700]" />}
                        <span className="shrink-0 text-[10px] text-gray-600">{formatDate(item.date)}</span>
                      </button>
                      <div className="absolute right-1 z-10 flex items-center gap-0.5 rounded-lg bg-[#21120b] p-0.5 md:hidden md:group-hover:flex md:group-focus-within:flex">
                        <button type="button" onClick={() => togglePin(item)} className="rounded p-1 text-gray-500 hover:bg-white/10 hover:text-[#ff812b]" title={item.pinned ? 'Unpin chat' : 'Pin chat'}><Pin className={`h-3.5 w-3.5 ${item.pinned ? 'fill-[#ff6700] text-[#ff6700]' : ''}`} /></button>
                        <button type="button" onClick={() => { setEditingHistoryId(item.id); setEditedTitle(item.title); }} className="rounded p-1 text-gray-500 hover:bg-white/10 hover:text-gray-200" title="Rename chat"><Pencil className="h-3.5 w-3.5" /></button>
                        <button type="button" onClick={() => deleteHistoryItem(item)} className="rounded p-1 text-gray-500 hover:bg-red-500/10 hover:text-red-300" title={item.type === 'server' ? 'Hide saved chat' : 'Delete local chat'}><Trash2 className="h-3.5 w-3.5" /></button>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
            {historyItems.length === 0 && <p className="px-3 py-2 text-sm text-gray-600">No saved chats yet.</p>}
            {visibleHistoryCount < historyItems.length && <button type="button" onClick={() => setVisibleHistoryCount((count) => count + HISTORY_PAGE_SIZE)} className="mt-2 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-gray-400 transition-colors hover:bg-[#21120b] hover:text-gray-200"><ChevronDown className="h-4 w-4" />Show more ({historyItems.length - visibleHistoryCount})</button>}
          </div>
        </div>
      </aside>

      <div className="relative flex min-w-0 flex-1 flex-col bg-gradient-to-br from-[#2a1104] via-[#160a03] to-black">
        <header className="absolute left-0 top-0 z-10 flex w-full items-start justify-between p-4">
          <div className="flex items-center gap-2">{!isSidebarOpen && <button type="button" onClick={() => setIsSidebarOpen(true)} className="flex items-center gap-2 rounded-full p-2 transition-colors hover:bg-white/5" title="Open sidebar"><img src="/logo.png" alt="Open sidebar" className="h-6 w-6 object-contain" /></button>}</div>
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => { setSelectedSyncId(''); setIsDeviceHubOpen(true); }}
              className="flex items-center gap-2 rounded-lg border border-[#ff6700]/30 bg-[#1f110a]/90 px-3 py-1.5 text-xs font-mono font-medium text-orange-200 backdrop-blur-sm transition-all hover:bg-[#ff6700]/20 hover:border-[#ff6700]/60"
              title="Hotspot Port Forwarding & Multi-Device Sync Hub"
            >
              <Radio className="h-3.5 w-3.5 text-[#ff6700] animate-pulse" />
              <span>Device Hub {latestSyncId ? `#${latestSyncId}` : ''}</span>
            </button>
            <StatusPill apiStatus={apiStatus} isOnline={isOnline} />
          </div>
        </header>


        <main className={`flex min-h-0 flex-1 flex-col overflow-hidden px-4 pt-16 ${hasMessages ? '' : 'justify-center pb-[10vh]'}`}>
          {hasMessages ? (
            <div ref={scrollContainerRef} className="min-h-0 flex-1 overflow-y-auto overscroll-contain pb-4 pt-5"><div className="mx-auto flex w-full max-w-[820px] flex-col gap-5">
              {activeChat.messages.map((message) => message.role === 'user' ? <UserMessage key={message.id} message={message} copiedId={copiedId} onCopied={setCopiedId} /> : <AssistantMessage key={message.id} message={message} copiedId={copiedId} onCopied={setCopiedId} onShareSync={(pin) => { setSelectedSyncId(pin); setIsDeviceHubOpen(true); }} />)}
              <div ref={messageEndRef} />
            </div></div>
          ) : (
            <div className="mb-6 flex flex-col items-center">
  <h1 className="mb-4 text-center text-[clamp(2rem,5vw,2.75rem)] font-normal tracking-tight text-gray-200">
    {typedHeadingCount <= HEADING_PREFIX.length
      ? HEADING_PREFIX.slice(0, typedHeadingCount)
      : (
        <>
          {HEADING_PREFIX}
          <span className="font-medium text-[#ff6700]">
            {HEADING_HIGHLIGHT.slice(0, Math.min(typedHeadingCount - HEADING_PREFIX.length, HEADING_HIGHLIGHT.length))}
          </span>
          {typedHeadingCount > HEADING_PREFIX.length + HEADING_HIGHLIGHT.length
            ? HEADING_SUFFIX.slice(0, typedHeadingCount - HEADING_PREFIX.length - HEADING_HIGHLIGHT.length)
            : ''}
        </>
      )}
    <span className={`ml-0.5 inline-block h-[1em] w-[2px] translate-y-[2px] bg-gray-400 align-middle ${typedHeadingCount >= HEADING_PREFIX.length + HEADING_HIGHLIGHT.length + HEADING_SUFFIX.length ? 'opacity-0' : 'animate-pulse'}`} />
  </h1>
</div>
          )}

          <div className={`mx-auto w-full max-w-[820px] ${hasMessages ? 'pb-5 pt-2' : ''}`}>
            {attachmentNotice && <div className="mb-2 flex items-start gap-2 rounded-xl border border-amber-400/15 bg-amber-400/10 px-3 py-2 text-xs leading-5 text-amber-100"><AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span>{attachmentNotice}</span><button type="button" onClick={() => setAttachmentNotice('')} className="ml-auto text-amber-200"><X className="h-3.5 w-3.5" /></button></div>}
            {attachments.length > 0 && <div className="mb-2 flex flex-wrap gap-2">{attachments.map((file, index) => <span key={`${file.name}-${file.lastModified}`} className="inline-flex max-w-full items-center gap-2 rounded-lg border border-[#ff6700]/20 bg-black/60 px-2.5 py-1.5 text-xs text-gray-300"><Paperclip className="h-3.5 w-3.5 text-[#ff6700]" /><span className="max-w-52 truncate">{file.name}</span><button type="button" onClick={() => setAttachments((current) => current.filter((_, fileIndex) => fileIndex !== index))} className="rounded text-gray-500 hover:text-red-300" title={`Remove ${file.name}`}><X className="h-3.5 w-3.5" /></button></span>)}</div>}

            <div className="relative flex w-full items-end gap-2 rounded-3xl border border-[#ff6700]/10 bg-black px-3 py-2 shadow-2xl shadow-[#ff6700]/5">
             {mode === 'agent' && <div ref={menuRef} className="relative mb-1 flex items-center gap-1">
  {activeFilter ? (
    <button type="button" onClick={() => setActiveFilter(null)} className="group relative shrink-0 rounded-full p-2 text-red-400 transition-colors hover:bg-[#ff6700]/10" title={`Remove ${DOC_TYPES[activeFilter].label} mode`}>
      {(() => {
        const ActiveIcon = DOC_TYPES[activeFilter].icon;
        return <ActiveIcon className="h-5 w-5 transition-opacity group-hover:opacity-0" />;
      })()}
      <X className="absolute inset-0 m-auto h-5 w-5 opacity-0 transition-opacity group-hover:opacity-100" />
    </button>
  ) : (
    <button type="button" onClick={() => setIsMenuOpen((open) => !open)} className="shrink-0 rounded-full p-2 text-gray-400 transition-colors hover:bg-[#ff6700]/10 hover:text-gray-200" title="Add files to Agent mode"><Plus className="h-5 w-5" /></button>
  )}
  {isMenuOpen && <div className="absolute bottom-12 left-0 z-50 w-max rounded-xl border border-[#ff6700]/20 bg-[#1f110a] py-1.5 shadow-2xl"><button type="button" onClick={() => { setIsMenuOpen(false); fileInputRef.current?.click(); }} className="flex items-center gap-3 px-4 py-2 text-left text-sm font-medium text-gray-300 transition-colors hover:bg-[#ff6700]/15 hover:text-white"><Upload className="h-4 w-4 text-[#ff6700]" />Upload files for RAG</button></div>}
</div>}
              <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" multiple />
              <textarea ref={textareaRef} value={input} onChange={handleInput} onKeyDown={(event) => { if (event.nativeEvent.isComposing) return; if ((event.key === 'Enter' && !event.shiftKey) || (event.key === 'Enter' && (event.ctrlKey || event.metaKey))) { event.preventDefault(); void handleSend(); } }} placeholder="What's the mission today?" rows={1} className="max-h-[200px] min-h-[40px] flex-1 resize-none overflow-y-auto bg-transparent px-2 py-2 text-[15px] leading-relaxed text-gray-100 placeholder-gray-600 outline-none" />
              <div className="mb-1 flex shrink-0 items-center gap-1.5">
                <div ref={modeMenuRef} className="relative"><button type="button" onClick={() => setIsModeMenuOpen((open) => !open)} className="flex items-center gap-1.5 rounded-full border border-white/[0.05] bg-[#24130b] px-3 py-2 text-xs font-medium text-gray-200 transition-colors hover:bg-[#2a160d]" title="Choose model mode">{mode === 'agent' ? 'Agent' : 'Chief'}<ChevronDown className={`h-3.5 w-3.5 text-gray-400 transition-transform ${isModeMenuOpen ? 'rotate-180' : ''}`} /></button>{isModeMenuOpen && <div className="absolute bottom-11 right-0 z-50 w-72 overflow-hidden rounded-xl border border-[#ff6700]/20 bg-black py-1.5 shadow-2xl"><button type="button" onClick={() => selectMode('agent')} className={`w-full px-4 py-2.5 text-left text-sm ${mode === 'agent' ? 'bg-[#ff6700]/10 text-[#ff812b]' : 'text-gray-300 hover:bg-[#ff6700]/10 hover:text-white'}`}><span className="block font-medium">Agent</span><span className="mt-0.5 block text-xs text-gray-500">Smart routing: Coder, Qwen 7B, Qwen VL and RAG tools.</span></button><button type="button" onClick={() => selectMode('chief')} className={`w-full px-4 py-2.5 text-left text-sm ${mode === 'chief' ? 'bg-[#ff6700]/10 text-[#ff812b]' : 'text-gray-300 hover:bg-[#ff6700]/10 hover:text-white'}`}><span className="block font-medium">Chief</span><span className="mt-0.5 block text-xs text-gray-500">Sends only this task to Qwen 14B complex mode.</span></button></div>}</div>
                {isProcessing ? <button type="button" onClick={handleStop} className="rounded-full bg-red-500/20 p-2.5 text-red-300 transition-colors hover:bg-red-500/30" title="Stop generation"><Square className="h-4 w-4 fill-current" /></button> : <button type="button" onClick={() => void handleSend()} disabled={!input.trim()} className="rounded-full bg-[#ff6700]/20 p-2.5 text-[#ff812b] transition-colors hover:bg-[#ff6700]/30 disabled:cursor-not-allowed disabled:opacity-35" title="Send message"><Send className="h-5 w-5" /></button>}
              </div>
            </div>

            <div className="mt-3 flex flex-wrap items-center justify-center gap-2 px-2">
              {Object.entries(DOC_TYPES).map(([type, document]) => {
                const Icon = document.icon;
                const selected = activeFilter === type;
                return <button key={type} type="button" disabled={mode !== 'agent'} onClick={() => setActiveFilter((current) => current === type ? null : type)} className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium transition-all ${selected ? 'border-[#ff6700] bg-[#ff6700]/20 text-[#ff812b]' : 'border-white/[0.04] bg-black/60 text-gray-400 hover:bg-[#1a0e08] hover:text-gray-200'} disabled:cursor-not-allowed disabled:opacity-35`} title={mode === 'agent' ? `Create a ${document.label} from a short topic` : 'Document creation is available in Agent mode'}><Icon className={`h-3.5 w-3.5 ${document.accent}`} />{document.label}</button>;
              })}
            </div>
            <div className="mt-3 flex justify-center gap-4 text-[11px] text-gray-600"><span><Keyboard className="mr-1 inline h-3 w-3" />Enter send · Shift Enter line break</span><span>Ctrl / focus</span><span>Ctrl Shift M switch mode</span></div>
          </div>
        </main>
      </div>

      {isSearchOpen && <div className="fixed inset-0 z-[60] flex items-start justify-center bg-black/70 px-4 pt-[12vh] backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="Search chat history"><div className="w-full max-w-xl overflow-hidden rounded-2xl border border-white/10 bg-[#17100c] shadow-2xl"><div className="flex items-center gap-3 border-b border-white/5 px-4 py-3"><Search className="h-5 w-5 text-gray-500" /><input ref={searchInputRef} value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} placeholder="Search by chat title…" className="flex-1 bg-transparent text-sm text-white outline-none placeholder:text-gray-600" /><button type="button" onClick={() => setIsSearchOpen(false)} className="rounded p-1 text-gray-500 hover:bg-white/5 hover:text-white"><X className="h-4 w-4" /></button></div><div className="max-h-80 overflow-y-auto p-2">{matchingHistory.length ? matchingHistory.map((item) => <button key={item.id} type="button" onClick={() => void openHistoryItem(item)} className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm text-gray-300 hover:bg-[#25150d]"><MessageSquare className="h-4 w-4 text-[#ff6700]" /><span className="min-w-0 flex-1 truncate">{item.title}</span><span className="text-xs text-gray-600">{formatDate(item.date)}</span></button>) : <p className="px-3 py-8 text-center text-sm text-gray-500">No chat titles match “{searchTerm}”.</p>}</div></div></div>}

      {showClearConfirm && <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="Clear history confirmation"><div className="w-full max-w-md rounded-2xl border border-red-400/20 bg-[#17100c] p-5 shadow-2xl"><div className="mb-3 flex items-center gap-2 text-red-300"><AlertTriangle className="h-5 w-5" /><h2 className="font-medium">Clear chat history?</h2></div><p className="text-sm leading-6 text-gray-400">This clears the visible frontend history and resets the backend’s active conversation memory. The current backend does not expose a database-delete endpoint, so its saved session records remain on disk.</p><div className="mt-5 flex justify-end gap-2"><button type="button" onClick={() => setShowClearConfirm(false)} className="rounded-xl px-3 py-2 text-sm text-gray-300 hover:bg-white/5">Cancel</button><button type="button" onClick={() => void clearVisibleHistory()} className="rounded-xl bg-red-500/15 px-3 py-2 text-sm font-medium text-red-200 hover:bg-red-500/25">Clear visible history</button></div></div></div>}

      {isSettingsOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="Application Settings">
          <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#17100c] p-6 shadow-2xl">
            <div className="mb-4 flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center gap-2 text-lg font-medium text-gray-200">
                <Settings className="h-5 w-5 text-[#ff6700]" />
                <span>Settings</span>
              </div>
              <button type="button" onClick={() => setIsSettingsOpen(false)} className="rounded p-1 text-gray-400 hover:bg-white/10 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-5">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-200">Open new chat on restart</p>
                  <p className="mt-0.5 text-xs leading-relaxed text-gray-400">When enabled, starting or reloading the app opens a new chat instead of continuing the previous chat.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setAutoNewChatOnStart((prev) => !prev)}
                  className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${autoNewChatOnStart ? 'bg-[#ff6700]' : 'bg-white/10'}`}
                  role="switch"
                  aria-checked={autoNewChatOnStart}
                >
                  <span className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${autoNewChatOnStart ? 'translate-x-5' : 'translate-x-0'}`} />
                </button>
              </div>

              <div className="border-t border-white/10 pt-4">
                <div className="mb-2">
                  <p className="text-sm font-medium text-gray-200">RAG Knowledge Base</p>
                  <p className="mt-0.5 text-xs leading-relaxed text-gray-400">Remove all uploaded files and purge vector database embeddings.</p>
                </div>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await clearKnowledgeBase();
                      setRagMemory([]);
                      writeStoredJson('agent-otg-rag-memory-v1', []);
                      setAttachmentNotice('RAG knowledge base cleared successfully.');
                      setIsSettingsOpen(false);
                    } catch (err) {
                      setAttachmentNotice(`Failed to clear RAG: ${err.message}`);
                    }
                  }}
                  className="mt-2 flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs font-medium text-red-300 hover:bg-red-500/20"
                >
                  <Trash2 className="h-4 w-4 text-red-400" />
                  Clear all RAG files
                </button>
              </div>
            </div>

            <div className="mt-6 flex justify-end">
              <button type="button" onClick={() => setIsSettingsOpen(false)} className="rounded-xl bg-[#ff6700]/20 px-4 py-2 text-sm font-medium text-[#ff812b] hover:bg-[#ff6700]/30">
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {isDeviceHubOpen && (
        <DeviceSyncModal
          isOpen={isDeviceHubOpen}
          onClose={() => setIsDeviceHubOpen(false)}
          initialSyncId={selectedSyncId}
        />
      )}
    </div>
  );
}
