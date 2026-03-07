import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import Orb from './Orb';
import './index.css';

type Message    = { role: 'user' | 'ai'; text: string; isDiff?: boolean; imageData?: string };
type OrbMode    = 'idle' | 'querying' | 'agent-processing';
type ActiveMode = 'query' | 'agent' | 'tutor';
type TestCase   = { id: number; input: string; expected: string };
type TestResult = { input: string; expected: string; actual: string | null; passed: boolean; error: string | null };
type ChatSession = { session_id: string; title: string; created_at?: string; turn_count?: number };

function App() {
  const [mode, setMode] = useState<OrbMode>('idle');
  const [activeMode, setActiveMode] = useState<ActiveMode>('query');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [queryMode, setQueryMode] = useState<'auto'|'fast'|'deep'|'deep_semantic'|'both'>('auto');
  const [agentTaskMode, setAgentTaskMode] = useState<'auto' | 'fix' | 'solve'>('auto');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [lastUserQuery, setLastUserQuery] = useState<string | null>(null);
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [pendingAgentEdit, setPendingAgentEdit] = useState<{ instruction: string; output: string; filePath: string; newSource?: string; contextEdits?: { path: string; diff: string; new_source: string; explanation?: string }[] } | null>(null);
  const computeSize = () => {
    const maxDim = Math.max(window.innerWidth, window.innerHeight);
    const raw = 2000 / maxDim;           // 2000 is arbitrary scale factor
    return Math.min(Math.max(raw, 200), 1200); // clamp between 200 and 1200px
  };
  const [baseSize, setBaseSize] = useState(computeSize());
  const [detailsContent, setDetailsContent] = useState<string | null>(null);
  const [ragChunkPrompt, setRagChunkPrompt] = useState<{ show: boolean; instruction: string; filePath: string } | null>(null);
  const [customChunkInput, setCustomChunkInput] = useState('');
  const [ragSearchMethod, setRagSearchMethod] = useState<'bm25' | 'semantic' | 'both'>('both');
  // ── Test runner state ──────────────────────────────────────────────────────
  const [testCases, setTestCases]         = useState<TestCase[]>([]);
  const [testResults, setTestResults]     = useState<TestResult[]>([]);
  const [showTestPanel, setShowTestPanel] = useState(false);
  const [testNextId, setTestNextId]       = useState(1);
  const lastAgentInstruction              = useRef<string>('');
  // ── Session management ─────────────────────────────────────────────────────
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID());
  const [showSidebar, setShowSidebar] = useState(false);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const sessionCreatedRef = useRef(false);  // track if current session was persisted
  const [useMultiAgent, setUseMultiAgent] = useState(false);  // multi-agent orchestration toggle
  // ── Multi-file context ─────────────────────────────────────────────────────
  const [contextFiles, setContextFiles] = useState<string[]>([]);  // additional files for cross-file context
  // ── Infinite scroll / pagination ───────────────────────────────────────────
  const MESSAGES_PER_PAGE = 50;
  const [visibleCount, setVisibleCount] = useState(MESSAGES_PER_PAGE);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  // ── Voice IO ───────────────────────────────────────────────────────────────
  const [isRecording, setIsRecording] = useState(false);
  const [voiceLoading, setVoiceLoading] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  // ── OCR ────────────────────────────────────────────────────────────────────
  const [ocrLoading, setOcrLoading] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<string | null>(null);
  const [showOcrMenu, setShowOcrMenu] = useState(false);
  const [pendingImageBlob, setPendingImageBlob] = useState<Blob | null>(null);
  const [pendingImageDataUrl, setPendingImageDataUrl] = useState<string | null>(null);

  // ── Tutor mode state ───────────────────────────────────────────────────────
  type TutorLesson = {
    title: string; explanation: string; rules: string[]; example_code: string;
    example_explanation: string; key_terms: string[];
  };
  type TutorProblem = {
    session_id: string; style: string; question: string; language: string;
    options?: string[]; test_cases?: { input: string; expected: string }[];
    function_name?: string; code_snippet?: string; lesson?: TutorLesson;
  };
  type TutorFeedback = { correct: boolean; feedback: string; solved: boolean; attempts: number; score?: number; missing_points?: string[] };
  const [tutorTopic, setTutorTopic] = useState('arrays');
  const [tutorDifficulty, setTutorDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [tutorLanguage, setTutorLanguage] = useState('python');
  const [tutorStyle, setTutorStyle] = useState<'mcq' | 'free_text' | 'code'>('mcq');
  const [tutorProblem, setTutorProblem] = useState<TutorProblem | null>(null);
  const [tutorFeedback, setTutorFeedback] = useState<TutorFeedback | null>(null);
  const [tutorHint, setTutorHint] = useState<string | null>(null);
  const [tutorCode, setTutorCode] = useState('');
  const [tutorAnswer, setTutorAnswer] = useState('');
  const [tutorCodeResults, setTutorCodeResults] = useState<{ input: string; expected: string; actual: string | null; passed: boolean }[] | null>(null);
  const [tutorLoading, setTutorLoading] = useState(false);
  const [showLesson, setShowLesson] = useState(true);

  // ── Tools panel state ──────────────────────────────────────────────────────
  const [showToolsPanel, setShowToolsPanel] = useState(false);
  const [toolsOutput, setToolsOutput] = useState<string | null>(null);
  const [toolsLoading, setToolsLoading] = useState(false);

  // Type declaration for electronAPI exposed from preload
  const electronAPI = (window as any).electronAPI as { captureScreen: () => Promise<string | null>; sendCropResult: (rect: any) => void; openFileDialog: (opts?: { multiple?: boolean }) => Promise<string | string[] | null> } | undefined;


  // Helper: convert data URL to Blob without fetch (avoids "Failed to fetch" on large data URLs)
  const dataUrlToBlob = (dataUrl: string): Blob => {
    const [header, b64] = dataUrl.split(',');
    const mime = header.match(/:(.*?);/)?.[1] || 'image/png';
    const bytes = atob(b64);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    return new Blob([arr], { type: mime });
  };


  // update baseSize whenever window resizes so orb scales with window size
  useEffect(() => {
    const update = () => {
      setBaseSize(computeSize());
    };
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  // ── Load sessions from server on mount ───────────────────────────────────
  const loadSessions = async () => {
    try {
      const res = await fetch('http://localhost:8000/sessions');
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch { /* server not up yet */ }
  };
  useEffect(() => { loadSessions(); }, []);

  // ── Session helpers ─────────────────────────────────────────────────────
  const createServerSession = async (title: string): Promise<string> => {
    try {
      const res = await fetch('http://localhost:8000/sessions/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      });
      const data = await res.json();
      loadSessions();
      return data.session_id;
    } catch { return sessionId; }
  };

  const switchSession = async (sid: string) => {
    setSessionId(sid);
    setMessages([]);
    setVisibleCount(MESSAGES_PER_PAGE);
    setLastUserQuery(null);
    setPendingAgentEdit(null);
    setDetailsContent(null);
    sessionCreatedRef.current = true; // already exists on server
    // Load history from server
    try {
      const res = await fetch('http://localhost:8000/sessions/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid }),
      });
      const data = await res.json();
      const restored: Message[] = (data.turns || []).map((t: { role: string; content: string }) => ({
        role: t.role === 'User' ? 'user' as const : 'ai' as const,
        text: t.content,
      }));
      setMessages(restored);
    } catch { /* couldn't load history */ }
  };

  const handleNewChat = async () => {
    const newId = crypto.randomUUID();
    setSessionId(newId);
    setMessages([]);
    setVisibleCount(MESSAGES_PER_PAGE);
    setLastUserQuery(null);
    setPendingAgentEdit(null);
    setDetailsContent(null);
    sessionCreatedRef.current = false;
  };

  const handleDeleteSession = async (sid: string) => {
    try {
      await fetch('http://localhost:8000/sessions/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid }),
      });
      setSessions(prev => prev.filter(s => s.session_id !== sid));
      if (sid === sessionId) handleNewChat();
    } catch { /* ignore */ }
  };

  const handleRenameSession = async (sid: string, newTitle: string) => {
    if (!newTitle.trim()) return;
    try {
      await fetch('http://localhost:8000/sessions/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, title: newTitle.trim() }),
      });
      setSessions(prev => prev.map(s => s.session_id === sid ? { ...s, title: newTitle.trim() } : s));
    } catch { /* ignore */ }
    setEditingSessionId(null);
    setEditTitle('');
  };

  useEffect(() => {
    // Ensure visibleCount always includes the newest messages
    setVisibleCount(prev => Math.max(prev, Math.min(messages.length, MESSAGES_PER_PAGE)));
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── WAV encoder (no ffmpeg needed — pure PCM → WAV) ──────────────────────
  const encodeWAV = (samples: Float32Array, sampleRate: number): Blob => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeStr = (offset: number, str: string) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);          // subchunk size
    view.setUint16(20, 1, true);           // PCM
    view.setUint16(22, 1, true);           // mono
    view.setUint32(24, sampleRate, true);  // sample rate
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);           // block align
    view.setUint16(34, 16, true);          // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
  };

  // ── Voice recording (Web Audio API → WAV — no ffmpeg) ────────────────────
  const startVoiceRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const SAMPLE_RATE = 16000;
      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      const source = ctx.createMediaStreamSource(stream);
      // ScriptProcessor captures raw PCM floats
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      pcmChunksRef.current = [];
      processor.onaudioprocess = (e) => {
        pcmChunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      audioContextRef.current = ctx;
      audioStreamRef.current = stream;
      processorRef.current = processor;
      sourceRef.current = source;
      setIsRecording(true);
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'ai', text: `Microphone error: ${err.message}` }]);
    }
  };

  const stopVoiceRecording = async () => {
    setIsRecording(false);
    // Disconnect graph
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    audioStreamRef.current?.getTracks().forEach(t => t.stop());
    await audioContextRef.current?.close();
    // Merge PCM chunks
    const chunks = pcmChunksRef.current;
    if (!chunks.length) return;
    const totalLen = chunks.reduce((s, c) => s + c.length, 0);
    const merged = new Float32Array(totalLen);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }
    const wavBlob = encodeWAV(merged, 16000);
    setVoiceLoading(true);
    try {
      const form = new FormData();
      form.append('audio', wavBlob, 'recording.wav');
      const res = await fetch('http://localhost:8000/voice/transcribe', { method: 'POST', body: form });
      const data = await res.json();
      if (data.status === 'ok' && data.text) {
        setInput(prev => (prev ? prev + ' ' : '') + data.text.trim());
      } else {
        setMessages(prev => [...prev, { role: 'ai', text: `Voice transcription error: ${data.error || 'unknown'}` }]);
      }
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'ai', text: `Voice error: ${err.message}` }]);
    } finally {
      setVoiceLoading(false);
      pcmChunksRef.current = [];
    }
  };

  // ── OCR capture ────────────────────────────────────────────────────────────
  const handleOCRCapture = async (file: File) => {
    setOcrLoading(true);
    try {
      const imageData = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.readAsDataURL(file);
      });
      // Store the image as pending context so user can ask questions about it
      setPendingImageBlob(file);
      setPendingImageDataUrl(imageData);
      // Send to analyze endpoint (LLM reasons about the image, not just OCR text)
      const form = new FormData();
      form.append('image', file, file.name);
      const res = await fetch('http://localhost:8000/ocr/analyze', { method: 'POST', body: form });
      const data = await res.json();
      if (data.status === 'ok') {
        setMessages(prev => [...prev,
          { role: 'user', text: `[Screenshot uploaded — ask me anything about it]`, imageData },
          { role: 'ai',   text: data.analysis || `[OCR RESULT]\n${data.ocr_text}` },
        ]);
      } else {
        setMessages(prev => [...prev, { role: 'ai', text: `OCR error: ${data.error || 'unknown'}` }]);
      }
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'ai', text: `OCR error: ${err.message}` }]);
    } finally {
      setOcrLoading(false);
    }
  };

  const handleScreenCapture = async () => {
    setShowOcrMenu(false);
    setOcrLoading(true);
    try {
      if (!electronAPI) {
        setMessages(prev => [...prev, { role: 'ai', text: 'Screen capture is only available in the Electron app.' }]);
        setOcrLoading(false);
        return;
      }
      const dataUrl = await electronAPI.captureScreen();
      if (!dataUrl) {
        setMessages(prev => [...prev, { role: 'ai', text: 'Screen capture cancelled.' }]);
        setOcrLoading(false);
        return;
      }
      // Convert data URL to Blob and store as pending context
      const blob = dataUrlToBlob(dataUrl);
      setPendingImageBlob(blob);
      setPendingImageDataUrl(dataUrl);
      // Send to analyze endpoint (LLM reasons about the image, not just OCR text)
      const form = new FormData();
      form.append('image', blob, 'screenshot.png');
      const res = await fetch('http://localhost:8000/ocr/analyze', { method: 'POST', body: form });
      const data = await res.json();
      if (data.status === 'ok') {
        setMessages(prev => [...prev,
          { role: 'user', text: `[Screen Capture — select area captured]`, imageData: dataUrl },
          { role: 'ai',   text: data.analysis || `[OCR RESULT]\n${data.ocr_text}` },
        ]);
      } else {
        setMessages(prev => [...prev, { role: 'ai', text: `OCR error: ${data.error || 'unknown'}` }]);
      }
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'ai', text: `Screen capture error: ${err.message}` }]);
    } finally {
      setOcrLoading(false);
    }
  };

  // ── Tutor handlers ─────────────────────────────────────────────────────────
  const handleTutorGenerate = async () => {
    setTutorLoading(true);
    setTutorProblem(null); setTutorFeedback(null); setTutorHint(null);
    setTutorCode(''); setTutorAnswer(''); setTutorCodeResults(null);
    try {
      const res = await fetch('http://localhost:8000/tutor/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: tutorTopic, difficulty: tutorDifficulty, language: tutorLanguage, style: tutorStyle }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setTutorProblem(data as TutorProblem);
      } else {
        setMessages(prev => [...prev, { role: 'ai', text: `[TUTOR] Error: ${data.error || 'unknown'}` }]);
      }
    } catch { setMessages(prev => [...prev, { role: 'ai', text: '[TUTOR] Failed to connect to server.' }]); }
    setTutorLoading(false);
  };

  const handleTutorCheckAnswer = async (answer: string) => {
    if (!tutorProblem) return;
    setTutorLoading(true);
    try {
      const res = await fetch('http://localhost:8000/tutor/check', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: tutorProblem.session_id, answer }),
      });
      const data = await res.json();
      if (data.status === 'ok') setTutorFeedback(data as TutorFeedback);
    } catch { /* ignore */ }
    setTutorLoading(false);
  };

  const handleTutorRunCode = async () => {
    if (!tutorProblem) return;
    setTutorLoading(true);
    try {
      const res = await fetch('http://localhost:8000/tutor/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: tutorProblem.session_id, code: tutorCode }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setTutorCodeResults(data.results || []);
        if (data.solved) setTutorFeedback({ correct: true, feedback: 'All tests passed! Well done!', solved: true, attempts: data.attempts });
      }
    } catch { /* ignore */ }
    setTutorLoading(false);
  };

  const handleTutorHint = async () => {
    if (!tutorProblem) return;
    setTutorLoading(true);
    try {
      const res = await fetch('http://localhost:8000/tutor/hint', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: tutorProblem.session_id }),
      });
      const data = await res.json();
      if (data.status === 'ok') setTutorHint(`Hint ${data.hint_number}/${data.total_hints}: ${data.hint}`);
    } catch { /* ignore */ }
    setTutorLoading(false);
  };

  // ── Tools panel helper ─────────────────────────────────────────────────────
  const runTool = async (endpoint: string, body: Record<string, any> = {}) => {
    setToolsLoading(true);
    setToolsOutput(null);
    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setToolsOutput(JSON.stringify(data, null, 2));
    } catch (err: any) {
      setToolsOutput(`Error: ${err.message}`);
    }
    setToolsLoading(false);
  };

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!input.trim()) return;

  const userMsg = input.trim();
  setInput('');
  setMessages(prev => [...prev, { role: 'user', text: userMsg, imageData: pendingImageDataUrl || undefined }]);

  // If there's a pending image, send the question + image to the analyze endpoint
  if (pendingImageBlob) {
    const imageBlob = pendingImageBlob;
    const imageDataUrl = pendingImageDataUrl;
    setPendingImageBlob(null);
    setPendingImageDataUrl(null);

    setMode('querying');
    try {
      const form = new FormData();
      form.append('image', imageBlob, 'screenshot.png');
      form.append('question', userMsg);
      const res = await fetch('http://localhost:8000/ocr/analyze', { method: 'POST', body: form });
      const data = await res.json();
      if (data.status === 'ok') {
        setMessages(prev => [...prev, { role: 'ai', text: data.analysis }]);
      } else {
        setMessages(prev => [...prev, { role: 'ai', text: `Analysis error: ${data.error || 'unknown'}` }]);
      }
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'ai', text: `Error: ${err.message}` }]);
    } finally {
      setMode('idle');
    }
    return;
  }

  // Auto-create persistent session on first message
  let sid = sessionId;
  if (!sessionCreatedRef.current) {
    const title = userMsg.length > 40 ? userMsg.slice(0, 40) + '…' : userMsg;
    sid = await createServerSession(title);
    setSessionId(sid);
    sessionCreatedRef.current = true;
  }
  
  // Set mode based on active mode
  const orbMode = activeMode === 'agent' ? 'agent-processing' : 'querying';
  setMode(orbMode);

  try {
    if (activeMode === 'agent') {
      // Agent mode: send to code agent endpoint
      if (!selectedFilePath) {
        setMessages(prev => [...prev, { role: 'ai', text: 'Error: No file selected. Please select a file to edit.' }]);
        setMode('idle');
        return;
      }
      
      lastAgentInstruction.current = userMsg;
      const endpoint = useMultiAgent ? '/agent/orchestrate' : '/agent/edit';
      const label = useMultiAgent ? '[MULTI-AGENT] Planner → Agent → Critic pipeline running...' : '[CODE AGENT] Processing your code request...';
      setMessages(prev => [...prev, { role: 'ai', text: label }]);
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction: userMsg, file_path: selectedFilePath, task_mode: agentTaskMode, session_id: sid, context_files: contextFiles }),
      });
      const data = await res.json();

      if (useMultiAgent && (data.status === 'ok' || data.status === 'pending_review')) {
        // Multi-agent returns {diff, explanation, citations, plan, verdict, attempts, related_files, new_source, critic_feedback, discussion_log}
        const planSteps = data.plan?.steps?.map((s: string, i: number) => `  ${i + 1}. ${s}`).join('\n') || '';
        const relatedNote = data.related_files?.length > 0 ? `\n📎 Related files read: ${data.related_files.join(', ')}` : '';
        const orchestrateMsgs: Message[] = [];
        if (planSteps) orchestrateMsgs.push({ role: 'ai', text: `[PLAN]\n${planSteps}` });
        // Show agent discussion if agents debated the problem
        if (data.discussion_log?.length > 0) {
          const discussionText = data.discussion_log.join('\n\n');
          orchestrateMsgs.push({ role: 'ai', text: `[AGENT DISCUSSION]\n${discussionText}` });
        }
        if (data.diff) orchestrateMsgs.push({ role: 'ai', text: `[DRY RUN PREVIEW]\n\n${data.diff}`, isDiff: true });
        if (data.explanation) {
          const citationBlock = data.citations?.length > 0
            ? `\n\n Sources:\n${data.citations.map((c: string, i: number) => `  ${i + 1}. ${c}`).join('\n')}`
            : '';
          orchestrateMsgs.push({ role: 'ai', text: `[EXPLANATION]\n${data.explanation}${citationBlock}${relatedNote}` });
        }
        // Build verdict message with critic feedback for transparency
        let verdictText = `[VERDICT] ${data.verdict} — ${data.attempts} attempt(s)`;
        if (data.verdict === 'FAIL' && data.critic_feedback) {
          verdictText += `\n\n[CRITIC FEEDBACK]\n${data.critic_feedback}`;
        }
        orchestrateMsgs.push({ role: 'ai', text: verdictText });
        setMessages(prev => [...prev, ...orchestrateMsgs]);
        if (data.diff && data.file_path) {
          setPendingAgentEdit({ instruction: userMsg, output: data.diff, filePath: data.file_path, newSource: data.new_source, contextEdits: data.context_file_edits || [] });
          setSelectedFilePath(data.file_path);
        }
      } else if (data.status === 'pending_review') {
        // Show dry_run output and set pending edit; use resolved path returned by server
        const resolvedPath = data.file_path || selectedFilePath || '';
        setSelectedFilePath(resolvedPath);
        const newMsgs: Message[] = [{ role: 'ai', text: `[DRY RUN PREVIEW]\n\n${data.dry_run_output}`, isDiff: true }];
        // Append explanation + citations if the agent returned them
        if (data.explanation) {
          const citationBlock = data.citations && data.citations.length > 0
            ? `\n\n📚 Sources:\n${data.citations.map((c: string, i: number) => `  ${i + 1}. ${c}`).join('\n')}`
            : '';
          newMsgs.push({ role: 'ai', text: `[EXPLANATION]\n${data.explanation}${citationBlock}` });
        }
        setMessages(prev => [...prev, ...newMsgs]);
        setPendingAgentEdit({ instruction: userMsg, output: data.dry_run_output, filePath: resolvedPath, newSource: data.new_source });
      } else if (data.status === 'error') {
        setMessages(prev => [...prev, { role: 'ai', text: `[AGENT ERROR] ${data.message || data.error}` }]);
      } else {
        setMessages(prev => [...prev, { role: 'ai', text: data.result || data.error || 'Agent processing complete' }]);
      }
    } else {
      // Query mode: send to query endpoint
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg, mode: queryMode, session_id: sid }),
      });
      const data = await res.json();

      // store last user query for potential dislike feedback
      setLastUserQuery(userMsg);

      if (data.fast && data.deep) {
        // both mode – show concise answers + summaries, stash full details
        setMessages(prev => [
          ...prev,
          { role: 'ai', text: `[FAST] ${data.fast.answer}` },
          { role: 'ai', text: `[FAST SUMMARY] ${data.fast.summary}` },
          { role: 'ai', text: `[DEEP] ${data.deep.answer}` },
          { role: 'ai', text: `[DEEP SUMMARY] ${data.deep.summary}` },
        ]);
        setDetailsContent(`FAST CITATIONS:\n${data.fast.citations}\n\nFAST DETAILED:\n${data.fast.detailed}\n\nDEEP CITATIONS:\n${data.deep.citations}\n\nDEEP DETAILED:\n${data.deep.detailed}`);
      } else {
        // regular single-mode response: show answer + summary for deep mode
        setMessages(prev => [...prev, { role: 'ai', text: data.answer }]);
        if (queryMode === 'deep' || queryMode === 'deep_semantic') {
          setMessages(prev => [...prev, { role: 'ai', text: `[SUMMARY] ${data.summary}` }]);
          setDetailsContent(`CITATIONS:\n${data.citations}\n\nDETAILED:\n${data.detailed}`);
        }
      }
    }
  } catch (err) {
    setMessages(prev => [...prev, { role: 'ai', text: 'Error connecting to Aion.' }]);
  }

  setMode('idle');
};

const handleRunTests = async () => {
  if (!selectedFilePath || testCases.length === 0) return;
  setMode('agent-processing');
  try {
    const res = await fetch('http://localhost:8000/agent/run_tests', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_path: selectedFilePath,
        test_cases: testCases.map(tc => ({ input: tc.input, expected: tc.expected })),
      }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      setTestResults(data.results);
    } else {
      setMessages(prev => [...prev, { role: 'ai', text: `[TEST ERROR] ${data.error}` }]);
    }
  } catch {
    setMessages(prev => [...prev, { role: 'ai', text: 'Test run failed — is the server running?' }]);
  }
  setMode('idle');
};

const handleFixWithTests = async () => {
  if (!selectedFilePath || testCases.length === 0) return;
  const instruction = lastAgentInstruction.current;
  if (!instruction) {
    setMessages(prev => [...prev, { role: 'ai', text: '[TEST] Send the task description first so the agent knows what to implement.' }]);
    return;
  }
  setMode('agent-processing');
  setMessages(prev => [...prev, { role: 'ai', text: `[AUTO-FIX] Running tests and iterating... (up to 3 attempts)` }]);
  try {
    const res = await fetch('http://localhost:8000/agent/fix_with_tests', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_path: selectedFilePath,
        instruction,
        test_cases: testCases.map(tc => ({ input: tc.input, expected: tc.expected })),
        max_retries: 3,
        task_mode: agentTaskMode === 'auto' ? 'solve' : agentTaskMode,
        session_id: sessionId,
      }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      setTestResults(data.test_results);
      const passed = data.test_results.filter((r: TestResult) => r.passed).length;
      const total  = data.test_results.length;
      const label  = data.all_passed ? '✓ All tests pass' : `${passed}/${total} tests pass`;
      const fixMsgs: Message[] = [
        { role: 'ai', text: `[AUTO-FIX] ${label} after ${data.attempts} fix attempt(s).\n\n${data.diff || '(no diff)'}`, isDiff: !!data.diff },
      ];
      if (data.explanation) {
        const citationBlock = data.citations && data.citations.length > 0
          ? `\n\n📚 Sources:\n${data.citations.map((c: string, i: number) => `  ${i + 1}. ${c}`).join('\n')}`
          : '';
        fixMsgs.push({ role: 'ai', text: `[EXPLANATION]\n${data.explanation}${citationBlock}` });
      }
      setMessages(prev => [...prev, ...fixMsgs]);
      if (data.file_path) setSelectedFilePath(data.file_path);
    } else {
      setMessages(prev => [...prev, { role: 'ai', text: `[AUTO-FIX ERROR] ${data.error}` }]);
    }
  } catch {
    setMessages(prev => [...prev, { role: 'ai', text: 'Auto-fix failed — is the server running?' }]);
  }
  setMode('idle');
};

const handleRagChunkRetry = async (chunks: number) => {
  if (!ragChunkPrompt) return;
  
  setMode('agent-processing');
  setMessages(prev => [...prev, { role: 'ai', text: `[CODE AGENT] Retrying with ${chunks} RAG chunks (${ragSearchMethod})...` }]);
  
  try {
    const res = await fetch('http://localhost:8000/agent/edit_with_chunks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        instruction: ragChunkPrompt.instruction, 
        file_path: ragChunkPrompt.filePath,
        max_chunks: chunks,
        task_mode: agentTaskMode,
        search_method: ragSearchMethod
      }),
    });
    const data = await res.json();
    
    if (data.status === 'pending_review') {
      const resolvedPath = data.file_path || ragChunkPrompt.filePath;
      setSelectedFilePath(resolvedPath);
      const ragMsgs: Message[] = [{ role: 'ai', text: `[DRY RUN PREVIEW]\n\n${data.dry_run_output}`, isDiff: true }];
      if (data.explanation) {
        const citationBlock = data.citations && data.citations.length > 0
          ? `\n\n📚 Sources:\n${data.citations.map((c: string, i: number) => `  ${i + 1}. ${c}`).join('\n')}`
          : '';
        ragMsgs.push({ role: 'ai', text: `[EXPLANATION]\n${data.explanation}${citationBlock}` });
      }
      setMessages(prev => [...prev, ...ragMsgs]);
      setPendingAgentEdit({ instruction: ragChunkPrompt.instruction, output: data.dry_run_output, filePath: resolvedPath, newSource: data.new_source });
    } else if (data.status === 'error') {
      setMessages(prev => [...prev, { role: 'ai', text: `[AGENT ERROR] ${data.message || data.error}` }]);
    }
  } catch (err) {
    setMessages(prev => [...prev, { role: 'ai', text: 'Error retrying with custom chunks.' }]);
  }
  
  setRagChunkPrompt(null);
  setCustomChunkInput('');
  setMode('idle');
};


return (
  <div
    style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: '#000',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'sans-serif',
      color: '#fff',
      overflow: 'hidden',
      gap: '24px',
      padding: '20px',
    }}
    >
      {/* ── Session Sidebar ─────────────────────────────────────────────── */}
      <div
        style={{
          position: 'fixed', top: 0, left: 0, bottom: 0,
          width: showSidebar ? '260px' : '0px',
          backgroundColor: 'rgba(10,10,15,0.98)',
          borderRight: showSidebar ? '1px solid rgba(85,51,255,0.3)' : 'none',
          zIndex: 80, overflowY: 'auto', overflowX: 'hidden',
          transition: 'width 0.25s ease',
          display: 'flex', flexDirection: 'column',
        }}
      >
        {showSidebar && (
          <>
            <div style={{ padding: '16px 14px 10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: '#5533ff', fontWeight: 'bold', fontSize: '13px', letterSpacing: '0.08em' }}>CHATS</span>
              <button onClick={handleNewChat}
                style={{ background: 'none', border: '1px dashed #5533ff55', color: '#5533ff', padding: '4px 10px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' }}>
                + New
              </button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px 12px' }}>
              {sessions.map(s => (
                <div key={s.session_id}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '8px 10px', marginBottom: '4px', borderRadius: '8px', cursor: 'pointer',
                    backgroundColor: s.session_id === sessionId ? 'rgba(85,51,255,0.2)' : 'transparent',
                    border: s.session_id === sessionId ? '1px solid rgba(85,51,255,0.4)' : '1px solid transparent',
                  }}
                  onClick={() => { if (editingSessionId !== s.session_id) switchSession(s.session_id); }}
                >
                  {editingSessionId === s.session_id ? (
                    <input
                      autoFocus
                      value={editTitle}
                      onChange={e => setEditTitle(e.target.value)}
                      onBlur={() => handleRenameSession(s.session_id, editTitle)}
                      onKeyDown={e => { if (e.key === 'Enter') handleRenameSession(s.session_id, editTitle); if (e.key === 'Escape') { setEditingSessionId(null); setEditTitle(''); } }}
                      style={{ flex: 1, padding: '2px 6px', backgroundColor: '#111', border: '1px solid #555', borderRadius: '4px', color: '#fff', fontSize: '12px', outline: 'none' }}
                      onClick={e => e.stopPropagation()}
                    />
                  ) : (
                    <span style={{ flex: 1, fontSize: '12px', color: s.session_id === sessionId ? '#fff' : '#aaa', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {s.title || 'Untitled'}
                    </span>
                  )}
                  <button onClick={e => { e.stopPropagation(); setEditingSessionId(s.session_id); setEditTitle(s.title || ''); }}
                    style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer', fontSize: '11px', padding: '2px' }}
                    title="Rename">✏</button>
                  <button onClick={e => { e.stopPropagation(); handleDeleteSession(s.session_id); }}
                    style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: '11px', padding: '2px' }}
                    title="Delete">🗑</button>
                </div>
              ))}
              {sessions.length === 0 && (
                <div style={{ color: '#555', fontSize: '12px', textAlign: 'center', paddingTop: '20px' }}>No saved chats yet</div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Sidebar toggle button */}
      <button
        onClick={() => setShowSidebar(prev => !prev)}
        style={{
          position: 'fixed', top: '50%', left: showSidebar ? '260px' : '0px',
          transform: 'translateY(-50%)',
          zIndex: 81, padding: '8px 4px', borderRadius: '0 8px 8px 0',
          backgroundColor: 'rgba(85,51,255,0.3)', border: 'none', color: '#fff',
          cursor: 'pointer', fontSize: '14px', transition: 'left 0.25s ease',
          backdropFilter: 'blur(10px)',
        }}
        title={showSidebar ? 'Hide sessions' : 'Show sessions'}
      >
        {showSidebar ? '◀' : '▶'}
      </button>

      {/* Top spacer (for nicer vertical balance) */}
      <div style={{ flex: 1 }} />

      {/* ── Top Left Toolbar: RAG / Data controls ──────────────────────── */}
      <div style={{ position: 'fixed', top: '20px', left: '20px', zIndex: 50, display: 'flex', gap: '8px' }}>
        <button
          onClick={async () => {
            try {
              const res = await fetch('http://localhost:8000/open_data_folder', { method: 'POST' });
              const json = await res.json();
              if (json.status === 'opened') {
                setMessages(prev => [...prev, { role: 'ai', text: `Opened data folder: ${json.path}` }]);
              } else {
                setMessages(prev => [...prev, { role: 'ai', text: `Open failed: ${json.error}` }]);
              }
            } catch (e) {
              setMessages(prev => [...prev, { role: 'ai', text: 'Open data folder failed.' }]);
            }
          }}
          style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#333', color: '#fff', border: 'none' }}
          disabled={mode !== 'idle'}
        >
          Open Data Folder
        </button>

        <button
          onClick={async () => {
            setMode('querying');
            setMessages(prev => [...prev, { role: 'user', text: 'Batch ingesting all PDFs in data folder...' }]);
            try {
              const res = await fetch('http://localhost:8000/ingest', { method: 'POST' });
              const json = await res.json();
              const topicCount = Object.keys(json.topics || {}).length;
              setMessages(prev => [...prev, { 
                role: 'ai', 
                text: `✓ Batch ingest complete! Processed all PDFs in data folder. Extracted ${topicCount} topics: ${Object.keys(json.topics || {}).slice(0, 10).join(', ')}${topicCount > 10 ? '...' : ''}` 
              }]);
            } catch (e) {
              setMessages(prev => [...prev, { role: 'ai', text: `Batch ingest failed: ${e.message}` }]);
            }
            setMode('idle');
          }}
          style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#2068b8', color: '#fff', border: 'none', fontWeight: 'bold' }}
          disabled={mode !== 'idle'}
          title="Process all PDF files in the data folder at once"
        >
          Batch Ingest All
        </button>

        <input id="pdf-upload" type="file" accept="application/pdf" style={{ display: 'none' }} onChange={async (e) => {
          const f = (e.target as HTMLInputElement).files?.[0];
          if (!f) return;
          setMode('querying');
          try {
            const anyF = f as any;
            if (anyF.path) {
              const res = await fetch('http://localhost:8000/ingest_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: anyF.path })
              });
              const j = await res.json();
              if (j.status === 'ingested') {
                setMessages(prev => [...prev, { role: 'ai', text: `Ingested ${j.filename || f.name}. Topics: ${j.topics?.join(', ') || ''}` }]);
              } else if (j.status === 'exists') {
                setMessages(prev => [...prev, { role: 'ai', text: `File already ingested: ${f.name}` }]);
              } else {
                setMessages(prev => [...prev, { role: 'ai', text: `Ingest failed: ${j.error || JSON.stringify(j)}` }]);
              }
            } else {
              const form = new FormData();
              form.append('file', f, f.name);
              const res = await fetch('http://localhost:8000/upload_and_ingest', { method: 'POST', body: form });
              const j = await res.json();
              if (j.status === 'ingested') {
                setMessages(prev => [...prev, { role: 'ai', text: `Uploaded and ingested ${j.filename}. Topics: ${j.topics?.join(', ') || ''}` }]);
              } else if (j.status === 'exists') {
                setMessages(prev => [...prev, { role: 'ai', text: `File already exists on server: ${j.filename}` }]);
              } else {
                setMessages(prev => [...prev, { role: 'ai', text: `Upload failed: ${j.error || JSON.stringify(j)}` }]);
              }
            }
          } catch (err) {
            setMessages(prev => [...prev, { role: 'ai', text: 'Upload/ingest failed.' }]);
          }
          setMode('idle');
          (document.getElementById('pdf-upload') as HTMLInputElement).value = '';
        }} />

        <button
          onClick={() => (document.getElementById('pdf-upload') as HTMLInputElement).click()}
          style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#228822', color: '#fff', border: 'none' }}
          disabled={mode !== 'idle'}
        >
          Upload PDF
        </button>

        {activeMode === 'query' && (
          <select
            value={queryMode}
            onChange={e => setQueryMode(e.target.value as any)}
            disabled={mode !== 'idle'}
            style={{ padding: '6px', borderRadius: '6px', backgroundColor: '#111', color: '#fff', border: '1px solid #333' }}
          >
            <option value="auto">Auto</option>
            <option value="fast">Fast</option>
            <option value="deep">Deep</option>
            <option value="deep_semantic">Deep (Semantic)</option>
            <option value="both">Both</option>
          </select>
        )}
      </div>

      {/* ── Tools Panel (slide-out) ────────────────────────────────────── */}
      <button
        onClick={() => setShowToolsPanel(p => !p)}
        style={{
          position: 'fixed', top: '20px', right: activeMode === 'agent' ? '320px' : '20px', zIndex: 51,
          padding: '6px 12px', borderRadius: '6px', border: '1px solid #555',
          backgroundColor: showToolsPanel ? 'rgba(85,51,255,0.3)' : '#222', color: '#fff',
          cursor: 'pointer', fontSize: '12px', transition: 'right 0.25s ease',
        }}
      >
        🔧 Tools
      </button>

      {showToolsPanel && (
        <div style={{
          position: 'fixed', top: '60px', right: '20px', zIndex: 50, width: '340px', maxHeight: '80vh',
          overflowY: 'auto', padding: '16px', borderRadius: '14px',
          backgroundColor: 'rgba(10,10,20,0.95)', border: '1px solid rgba(85,51,255,0.3)',
          backdropFilter: 'blur(20px)', boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        }}>
          <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#5533ff', marginBottom: '12px', letterSpacing: '0.08em' }}>DEV TOOLS</div>

          {/* Lint */}
          <div style={{ marginBottom: '10px' }}>
            <button onClick={() => { if (selectedFilePath) runTool('/tools/lint', { file_path: selectedFilePath }); }}
              disabled={toolsLoading || !selectedFilePath}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left' }}>
              🔍 Lint Current File
            </button>
          </div>

          {/* Lint + Fix */}
          <div style={{ marginBottom: '10px' }}>
            <button onClick={() => { if (selectedFilePath) runTool('/tools/lint', { file_path: selectedFilePath, fix: true }); }}
              disabled={toolsLoading || !selectedFilePath}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left' }}>
              🔧 Lint + Auto-fix
            </button>
          </div>

          {/* Type Check */}
          <div style={{ marginBottom: '10px' }}>
            <button onClick={() => { if (selectedFilePath) runTool('/tools/type_check', { file_path: selectedFilePath }); }}
              disabled={toolsLoading || !selectedFilePath}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left' }}>
              🏷 Type Check
            </button>
          </div>

          {/* Pytest */}
          <div style={{ marginBottom: '10px' }}>
            <button onClick={() => runTool('/tools/pytest', { target: '.', with_coverage: false })}
              disabled={toolsLoading}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left' }}>
              🧪 Run Pytest
            </button>
          </div>

          {/* Git Diff */}
          <div style={{ marginBottom: '10px' }}>
            <button onClick={() => runTool('/tools/git_diff', { ref: 'HEAD' })}
              disabled={toolsLoading}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left' }}>
              📝 Git Diff (unstaged)
            </button>
          </div>

          {/* Git Diff Staged */}
          <div style={{ marginBottom: '10px' }}>
            <button onClick={() => runTool('/tools/git_diff_staged', {})}
              disabled={toolsLoading}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left' }}>
              📋 Git Diff (staged)
            </button>
          </div>

          {/* Git Log */}
          <div style={{ marginBottom: '10px' }}>
            <button onClick={() => runTool('/tools/git_log', { count: 15 })}
              disabled={toolsLoading}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left' }}>
              📜 Git Log (recent)
            </button>
          </div>

          {/* Pre-commit */}
          <div style={{ marginBottom: '10px' }}>
            <button onClick={() => runTool('/tools/pre_commit', {})}
              disabled={toolsLoading}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', cursor: 'pointer', fontSize: '12px', textAlign: 'left' }}>
              ✅ Pre-commit Check
            </button>
          </div>

          {!selectedFilePath && (
            <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px' }}>
              Select a file in agent mode to enable file-specific tools.
            </div>
          )}

          {/* Output */}
          {toolsLoading && <div style={{ color: '#5533ff', fontSize: '12px', marginTop: '8px' }}>Running...</div>}
          {toolsOutput && (
            <pre style={{
              marginTop: '10px', padding: '10px', borderRadius: '8px', backgroundColor: '#0a0a0a',
              border: '1px solid #333', color: '#ccc', fontSize: '11px', fontFamily: 'monospace',
              whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '300px', overflowY: 'auto',
            }}>
              {toolsOutput}
            </pre>
          )}
        </div>
      )}

      {/* ── Top Right Toolbar: File + Context (agent mode only) ────────── */}
      {activeMode === 'agent' && (
        <div style={{ position: 'fixed', top: '20px', right: '20px', zIndex: 50, display: 'flex', flexDirection: 'column', gap: '6px', alignItems: 'flex-end' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button
              onClick={async () => {
                const api = (window as any).electronAPI;
                let handled = false;
                if (api?.openFileDialog) {
                  try {
                    const p = await api.openFileDialog({ multiple: false });
                    if (p) { setSelectedFilePath(p as string); setTestResults([]); }
                    handled = true;
                  } catch { /* handler not yet registered — fall through */ }
                }
                if (!handled) {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.onchange = () => {
                    const f = input.files?.[0];
                    if (f) { setSelectedFilePath((f as any).path || f.name); setTestResults([]); }
                  };
                  input.click();
                }
              }}
              style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#4422aa', color: '#fff', border: 'none', cursor: 'pointer', whiteSpace: 'nowrap' }}
              disabled={mode !== 'idle'}
            >
              Select File
            </button>
            <button
              type="button"
              onClick={async () => {
                const api = (window as any).electronAPI;
                let handled = false;
                if (api?.openFileDialog) {
                  try {
                    const paths = await api.openFileDialog({ multiple: true });
                    if (paths) {
                      const arr: string[] = Array.isArray(paths) ? paths : [paths];
                      setContextFiles(prev => {
                        const existing = new Set(prev);
                        const merged = [...prev];
                        for (const p of arr) { if (!existing.has(p) && p !== selectedFilePath) merged.push(p); }
                        return merged.slice(0, 10);
                      });
                    }
                    handled = true;
                  } catch { /* handler not yet registered — fall through */ }
                }
                if (!handled) {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.multiple = true;
                  input.onchange = () => {
                    const files = input.files;
                    if (!files) return;
                    const arr: string[] = [];
                    for (let i = 0; i < files.length; i++) {
                      const p = (files[i] as any).path || files[i].name;
                      if (p && p !== selectedFilePath) arr.push(p);
                    }
                    setContextFiles(prev => {
                      const existing = new Set(prev);
                      const merged = [...prev];
                      for (const p of arr) { if (!existing.has(p)) merged.push(p); }
                      return merged.slice(0, 10);
                    });
                  };
                  input.click();
                }
              }}
              disabled={mode !== 'idle'}
              title="Add context files for cross-file editing (up to 10)"
              style={{
                padding: '6px 10px', borderRadius: '6px', border: '1px solid',
                borderColor: contextFiles.length > 0 ? '#22ff88' : '#555',
                backgroundColor: contextFiles.length > 0 ? 'rgba(34,255,136,0.15)' : '#111',
                color: contextFiles.length > 0 ? '#22ff88' : '#aaa',
                fontSize: '12px', cursor: 'pointer', whiteSpace: 'nowrap'
              }}
            >
              📂 Context{contextFiles.length > 0 ? ` (${contextFiles.length})` : ''}
            </button>
          </div>

          {/* Selected file path dropdown */}
          {selectedFilePath && (
            <div style={{
              padding: '5px 10px', borderRadius: '8px',
              backgroundColor: 'rgba(68,34,170,0.2)', border: '1px solid rgba(68,34,170,0.4)',
              fontSize: '11px', color: '#cca8ff', maxWidth: '400px',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              display: 'flex', alignItems: 'center', gap: '6px',
            }}>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }} title={selectedFilePath}>
                {selectedFilePath}
              </span>
              <button
                onClick={() => { setSelectedFilePath(null); setTestResults([]); }}
                style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '11px', padding: '0 2px' }}
              >✕</button>
            </div>
          )}

          {/* Context files tags */}
          {contextFiles.length > 0 && (
            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', maxWidth: '400px', justifyContent: 'flex-end' }}>
              {contextFiles.map((cf, i) => (
                <span key={i} style={{
                  fontSize: '10px', padding: '3px 8px', borderRadius: '10px',
                  backgroundColor: 'rgba(34,255,136,0.12)', border: '1px solid rgba(34,255,136,0.3)',
                  color: '#22ff88', display: 'inline-flex', alignItems: 'center', gap: '4px',
                }}>
                  {cf.split(/[/\\]/).pop()}
                  <button onClick={() => setContextFiles(prev => prev.filter((_, j) => j !== i))}
                    style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '10px', padding: '0 2px' }}>✕</button>
                </span>
              ))}
              <button onClick={() => setContextFiles([])}
                style={{ fontSize: '10px', padding: '3px 6px', borderRadius: '10px', backgroundColor: 'rgba(255,50,50,0.15)', border: '1px solid rgba(255,50,50,0.3)', color: '#ff5555', cursor: 'pointer' }}>
                Clear
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── Test Panel ────────────────────────────────────────────────────── */}
      {showTestPanel && activeMode === 'agent' && (
        <div style={{
          position: 'fixed', top: '70px', left: '20px', zIndex: 60,
          width: '380px', maxHeight: 'calc(100vh - 110px)',
          backgroundColor: 'rgba(15,15,20,0.97)',
          border: '1px solid rgba(34,187,255,0.3)',
          borderRadius: '12px', padding: '16px',
          display: 'flex', flexDirection: 'column', gap: '10px',
          overflowY: 'auto',
          boxShadow: '0 8px 32px rgba(0,0,0,0.7)',
        }}>
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#22bbff', fontWeight: 'bold', fontSize: '13px', letterSpacing: '0.06em' }}>
              🧪 TEST CASES
            </span>
            <button onClick={() => setShowTestPanel(false)}
              style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer', fontSize: '16px' }}>✕</button>
          </div>

          {/* Test case list */}
          {testCases.map((tc, i) => {
            const result = testResults[i] ?? null;
            const statusColor = result ? (result.passed ? '#44ff88' : '#ff5555') : '#555';
            const statusIcon  = result ? (result.passed ? '✓' : '✗') : '○';
            return (
              <div key={tc.id} style={{
                padding: '10px', borderRadius: '8px',
                border: `1px solid ${statusColor}44`,
                backgroundColor: 'rgba(255,255,255,0.04)',
                display: 'flex', flexDirection: 'column', gap: '6px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ color: statusColor, fontSize: '12px', fontWeight: 'bold', minWidth: '16px' }}>{statusIcon}</span>
                  <span style={{ color: '#888', fontSize: '11px' }}>Test {i + 1}</span>
                  <button
                    onClick={() => setTestCases(prev => prev.filter(t => t.id !== tc.id))}
                    style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: '13px' }}
                  >🗑</button>
                </div>
                <textarea
                  placeholder="Input — e.g. [[1,0,0],[0,0,1],[1,0,0]]&#10;Separate multiple args with newlines"
                  value={tc.input}
                  onChange={e => setTestCases(prev => prev.map(t => t.id === tc.id ? { ...t, input: e.target.value } : t))}
                  rows={2}
                  style={{ padding: '5px 8px', borderRadius: '6px', backgroundColor: '#0d0d14', color: '#fff', border: '1px solid #333', fontSize: '12px', fontFamily: 'monospace', resize: 'vertical', minHeight: '36px' }}
                />
                <input
                  placeholder="Expected output (e.g. 3)"
                  value={tc.expected}
                  onChange={e => setTestCases(prev => prev.map(t => t.id === tc.id ? { ...t, expected: e.target.value } : t))}
                  style={{ padding: '5px 8px', borderRadius: '6px', backgroundColor: '#0d0d14', color: '#fff', border: '1px solid #333', fontSize: '12px', fontFamily: 'monospace' }}
                />
                {result && !result.passed && (
                  <div style={{ fontSize: '11px', color: '#ff8877', fontFamily: 'monospace', paddingTop: '2px' }}>
                    {result.error
                      ? `⚠ ${result.error.split('\n')[0]}`
                      : `got: ${result.actual ?? 'null'}`
                    }
                  </div>
                )}
              </div>
            );
          })}

          {/* Add test case */}
          <button
            onClick={() => {
              setTestCases(prev => [...prev, { id: testNextId, input: '', expected: '' }]);
              setTestNextId(n => n + 1);
            }}
            style={{ padding: '7px', borderRadius: '8px', backgroundColor: '#1a1a2e', color: '#22bbff', border: '1px dashed #22bbff44', cursor: 'pointer', fontSize: '12px' }}
          >
            + Add Test Case
          </button>

          {/* Action buttons */}
          <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
            <button
              onClick={handleRunTests}
              disabled={mode !== 'idle' || testCases.length === 0 || !selectedFilePath}
              style={{
                flex: 1, padding: '8px', borderRadius: '8px', border: 'none',
                backgroundColor: mode !== 'idle' ? '#222' : '#1166cc',
                color: '#fff', cursor: mode !== 'idle' ? 'not-allowed' : 'pointer', fontSize: '13px'
              }}
            >
              ▶ Run Tests
            </button>
            <button
              onClick={handleFixWithTests}
              disabled={mode !== 'idle' || testCases.length === 0 || !selectedFilePath || testResults.every(r => r.passed)}
              title={!lastAgentInstruction.current ? 'Send your task description first' : 'Auto-fix code using test failures'}
              style={{
                flex: 1, padding: '8px', borderRadius: '8px', border: 'none',
                backgroundColor: (mode !== 'idle' || testResults.every(r => r.passed)) ? '#222' : '#883300',
                color: '#fff',
                cursor: (mode !== 'idle' || testResults.every(r => r.passed)) ? 'not-allowed' : 'pointer',
                fontSize: '13px'
              }}
            >
              🔁 Auto-fix
            </button>
          </div>

          {/* Summary */}
          {testResults.length > 0 && (
            <div style={{ fontSize: '12px', textAlign: 'center', paddingTop: '4px',
              color: testResults.every(r => r.passed) ? '#44ff88' : '#ff8855' }}>
              {testResults.filter(r => r.passed).length}/{testResults.length} passing
            </div>
          )}
        </div>
      )}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          zIndex: 1,                        // keep behind UI
          pointerEvents: 'none',            // clicks pass through
          backgroundColor: 'rgba(0,0,0,0.1)', // light shade to confirm visibility
        }}
      >
        <Orb
          hue={
            mode === 'agent-processing' ? 270 :  // Magenta for agent processing
            mode === 'querying' ? 200 :           // Cyan for query processing
            0                                      // Red for idle
          }
          hoverIntensity={
            mode === 'agent-processing' ? 2.5 :
            mode === 'querying' ? 2 : 0.4
          }
          rotateOnHover={true}
          forceHoverState={mode !== 'idle'}
          backgroundColor="#000000"
        />
        <div
          style={{
            position: 'absolute',
            bottom: '-32px',
            left: '50%',
            transform: 'translateX(-50%)',
            fontSize: '11px',
            letterSpacing: '0.18em',
            color: 
              mode === 'agent-processing' ? '#ff00ff' :  // Magenta for agent
              mode === 'querying' ? '#7df9ff' :          // Cyan for query
              '#777',
            textTransform: 'uppercase',
            transition: 'color 0.4s',
          }}
        >
          {mode === 'agent-processing' ? 'Processing Code…' : mode === 'querying' ? 'Thinking…' : 'Idle'}
        </div>
      </div>

      {/* ── Tutor Mode Panel ─────────────────────────────────────────────── */}
      {activeMode === 'tutor' && (
        <div style={{
          position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          zIndex: 35, width: '85%', maxWidth: '1000px', maxHeight: '80vh', overflowY: 'auto',
          padding: '24px', borderRadius: '20px',
          backgroundColor: 'rgba(0, 20, 15, 0.92)', backdropFilter: 'blur(25px)',
          border: '1px solid rgba(0,204,136,0.3)', boxShadow: '0 12px 40px rgba(0,0,0,0.6)',
        }}>
          {/* Setup bar */}
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px', alignItems: 'center' }}>
            <input value={tutorTopic} onChange={e => setTutorTopic(e.target.value)} placeholder="Topic (e.g. arrays, recursion)"
              style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', flex: '1', minWidth: '120px' }} />
            <select value={tutorDifficulty} onChange={e => setTutorDifficulty(e.target.value as any)}
              style={{ padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px' }}>
              <option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option>
            </select>
            <select value={tutorLanguage} onChange={e => setTutorLanguage(e.target.value)}
              style={{ padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px' }}>
              {['python', 'go', 'cpp', 'c', 'javascript', 'typescript', 'java', 'rust'].map(l =>
                <option key={l} value={l}>{l}</option>
              )}
            </select>
            <select value={tutorStyle} onChange={e => setTutorStyle(e.target.value as any)}
              style={{ padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px' }}>
              <option value="mcq">Multiple Choice</option><option value="free_text">Short Answer</option><option value="code">Coding</option>
            </select>
            <button onClick={() => { handleTutorGenerate(); setShowLesson(true); }} disabled={tutorLoading || !tutorTopic.trim()}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', backgroundColor: '#00cc88', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '13px', opacity: tutorLoading ? 0.5 : 1 }}>
              {tutorLoading ? '...' : 'Generate Problem'}
            </button>
          </div>

          {/* Lesson display */}
          {tutorProblem?.lesson && showLesson && (
            <div style={{ marginBottom: '16px', padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(85,51,255,0.08)', border: '1px solid rgba(85,51,255,0.25)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#b388ff' }}>
                  📖 {tutorProblem.lesson.title || `${tutorTopic} Lesson`}
                </div>
                <button onClick={() => setShowLesson(false)} style={{ background: 'none', border: '1px solid #5533ff44', borderRadius: '6px', color: '#888', cursor: 'pointer', padding: '4px 10px', fontSize: '11px' }}>
                  Hide Lesson
                </button>
              </div>
              <div style={{ fontSize: '13px', lineHeight: '1.7', color: '#ddd', whiteSpace: 'pre-wrap', marginBottom: '12px' }}>
                {tutorProblem.lesson.explanation}
              </div>
              {tutorProblem.lesson.rules && tutorProblem.lesson.rules.length > 0 && (
                <div style={{ marginBottom: '12px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#00cc88', marginBottom: '6px' }}>Rules:</div>
                  {tutorProblem.lesson.rules.map((rule, i) => (
                    <div key={i} style={{ fontSize: '12px', color: '#ccc', padding: '4px 0 4px 12px', borderLeft: '2px solid #00cc8844' }}>
                      {rule}
                    </div>
                  ))}
                </div>
              )}
              {tutorProblem.lesson.example_code && (
                <div style={{ marginBottom: '10px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#7df9ff', marginBottom: '6px' }}>Example:</div>
                  <pre style={{ padding: '12px', borderRadius: '8px', backgroundColor: '#0a0a0a', color: '#0f0', fontFamily: 'monospace', fontSize: '12px', overflowX: 'auto', border: '1px solid #333', whiteSpace: 'pre-wrap' }}>
                    {tutorProblem.lesson.example_code}
                  </pre>
                  {tutorProblem.lesson.example_explanation && (
                    <div style={{ fontSize: '12px', color: '#aaa', marginTop: '6px', lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>
                      {tutorProblem.lesson.example_explanation}
                    </div>
                  )}
                </div>
              )}
              {tutorProblem.lesson.key_terms && tutorProblem.lesson.key_terms.length > 0 && (
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {tutorProblem.lesson.key_terms.map((term, i) => (
                    <span key={i} style={{ padding: '2px 8px', borderRadius: '4px', backgroundColor: 'rgba(0,204,136,0.15)', color: '#00cc88', fontSize: '11px' }}>{term}</span>
                  ))}
                </div>
              )}
            </div>
          )}
          {tutorProblem?.lesson && !showLesson && (
            <button onClick={() => setShowLesson(true)} style={{ marginBottom: '12px', background: 'none', border: '1px solid #5533ff44', borderRadius: '6px', color: '#b388ff', cursor: 'pointer', padding: '4px 12px', fontSize: '11px' }}>
              📖 Show Lesson
            </button>
          )}

          {/* Problem display */}
          {tutorProblem && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(255,255,255,0.06)', border: '1px solid rgba(0,204,136,0.2)', marginBottom: '12px' }}>
                <div style={{ fontSize: '14px', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>{tutorProblem.question}</div>
              </div>

              {/* MCQ code snippet — shown above the options */}
              {tutorProblem.style === 'mcq' && tutorProblem.code_snippet && (
                <div style={{ marginBottom: '12px' }}>
                  <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#7df9ff', marginBottom: '6px' }}>Code:</div>
                  <pre style={{
                    padding: '12px', borderRadius: '8px', backgroundColor: '#0a0a0a', color: '#0f0',
                    fontFamily: 'monospace', fontSize: '12px', overflowX: 'auto', border: '1px solid #333',
                    whiteSpace: 'pre-wrap', lineHeight: '1.5',
                  }}>
                    {tutorProblem.code_snippet}
                  </pre>
                </div>
              )}

              {/* MCQ options */}
              {tutorProblem.style === 'mcq' && tutorProblem.options && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '12px' }}>
                  {tutorProblem.options.map((opt, i) => (
                    <button key={i} onClick={() => handleTutorCheckAnswer(opt.charAt(0))}
                      disabled={tutorLoading || (tutorFeedback?.solved ?? false)}
                      style={{
                        padding: '10px 16px', borderRadius: '10px', border: '1px solid rgba(0,204,136,0.3)',
                        backgroundColor: tutorFeedback?.solved && opt.charAt(0) === tutorFeedback?.feedback?.charAt(0) ? 'rgba(0,204,136,0.2)' : 'rgba(255,255,255,0.04)',
                        color: '#fff', cursor: 'pointer', fontSize: '13px', textAlign: 'left', transition: 'background 0.2s',
                      }}>
                      {opt}
                    </button>
                  ))}
                </div>
              )}

              {/* Free text input */}
              {tutorProblem.style === 'free_text' && (
                <div style={{ marginBottom: '12px' }}>
                  <textarea value={tutorAnswer} onChange={e => setTutorAnswer(e.target.value)}
                    placeholder="Type your answer here..." rows={4}
                    style={{ width: '100%', padding: '12px', borderRadius: '10px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', resize: 'vertical', boxSizing: 'border-box' }} />
                  <button onClick={() => handleTutorCheckAnswer(tutorAnswer)}
                    disabled={tutorLoading || !tutorAnswer.trim() || (tutorFeedback?.solved ?? false)}
                    style={{ marginTop: '8px', padding: '8px 20px', borderRadius: '8px', border: 'none', backgroundColor: '#00cc88', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '13px' }}>
                    Submit Answer
                  </button>
                </div>
              )}

              {/* Code editor with syntax support */}
              {tutorProblem.style === 'code' && (
                <div style={{ marginBottom: '12px' }}>
                  {tutorProblem.test_cases && tutorProblem.test_cases.length > 0 && (
                    <div style={{ marginBottom: '10px', fontSize: '12px', color: '#aaa' }}>
                      <strong style={{ color: '#00cc88' }}>Test Cases:</strong>
                      {tutorProblem.test_cases.map((tc, i) => (
                        <div key={i} style={{ padding: '4px 0', borderBottom: '1px solid #222' }}>
                          Input: <code style={{ color: '#7df9ff' }}>{tc.input}</code> → Expected: <code style={{ color: '#7df9ff' }}>{tc.expected}</code>
                        </div>
                      ))}
                    </div>
                  )}
                  <textarea value={tutorCode}
                    onChange={e => setTutorCode(e.target.value)}
                    onKeyDown={e => {
                      const ta = e.target as HTMLTextAreaElement;
                      // Tab key: insert 4 spaces (or 1 tab) at cursor
                      if (e.key === 'Tab') {
                        e.preventDefault();
                        const start = ta.selectionStart;
                        const end = ta.selectionEnd;
                        const indent = '    ';
                        const newVal = tutorCode.substring(0, start) + indent + tutorCode.substring(end);
                        setTutorCode(newVal);
                        // Restore cursor after React re-render
                        requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = start + indent.length; });
                      }
                      // Auto-close brackets
                      const pairs: Record<string, string> = { '{': '}', '[': ']', '(': ')' };
                      if (pairs[e.key]) {
                        e.preventDefault();
                        const start = ta.selectionStart;
                        const end = ta.selectionEnd;
                        const selected = tutorCode.substring(start, end);
                        const newVal = tutorCode.substring(0, start) + e.key + selected + pairs[e.key] + tutorCode.substring(end);
                        setTutorCode(newVal);
                        requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = start + 1; });
                      }
                      // Auto-close quotes
                      if (e.key === '"' || e.key === "'") {
                        const start = ta.selectionStart;
                        const charBefore = start > 0 ? tutorCode[start - 1] : '';
                        // Don't auto-close if it's an escape or already in a pair
                        if (charBefore !== '\\') {
                          e.preventDefault();
                          const end = ta.selectionEnd;
                          const selected = tutorCode.substring(start, end);
                          const newVal = tutorCode.substring(0, start) + e.key + selected + e.key + tutorCode.substring(end);
                          setTutorCode(newVal);
                          requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = start + 1; });
                        }
                      }
                      // Enter: maintain indentation
                      if (e.key === 'Enter') {
                        const start = ta.selectionStart;
                        const lineStart = tutorCode.lastIndexOf('\n', start - 1) + 1;
                        const currentLine = tutorCode.substring(lineStart, start);
                        const indentMatch = currentLine.match(/^(\s*)/);
                        let indent = indentMatch ? indentMatch[1] : '';
                        // Add extra indent after { or :
                        const trimmed = currentLine.trimEnd();
                        if (trimmed.endsWith('{') || trimmed.endsWith(':') || trimmed.endsWith('(')) {
                          indent += '    ';
                        }
                        if (indent) {
                          e.preventDefault();
                          const newVal = tutorCode.substring(0, start) + '\n' + indent + tutorCode.substring(ta.selectionEnd);
                          setTutorCode(newVal);
                          requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = start + 1 + indent.length; });
                        }
                      }
                    }}
                    placeholder={`Write your ${tutorProblem.language} solution here...`} rows={12}
                    spellCheck={false}
                    style={{ width: '100%', padding: '12px', borderRadius: '10px', border: '1px solid #333', backgroundColor: '#0a0a0a', color: '#0f0', fontSize: '13px', fontFamily: "'Fira Code', 'JetBrains Mono', 'Cascadia Code', Consolas, monospace", resize: 'vertical', tabSize: 4, boxSizing: 'border-box', lineHeight: '1.5', letterSpacing: '0.02em' }} />
                  <button onClick={handleTutorRunCode}
                    disabled={tutorLoading || !tutorCode.trim()}
                    style={{ marginTop: '8px', padding: '8px 20px', borderRadius: '8px', border: 'none', backgroundColor: '#00cc88', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '13px' }}>
                    ▶ Run Code
                  </button>
                  {/* Code test results */}
                  {tutorCodeResults && (
                    <div style={{ marginTop: '10px' }}>
                      {tutorCodeResults.map((r, i) => (
                        <div key={i} style={{ padding: '6px 10px', marginBottom: '4px', borderRadius: '6px', fontSize: '12px', fontFamily: 'monospace', backgroundColor: r.passed ? 'rgba(0,204,136,0.15)' : 'rgba(255,50,50,0.15)', border: `1px solid ${r.passed ? '#00cc8844' : '#ff333344'}` }}>
                          {r.passed ? '✓' : '✗'} Input: {r.input} | Expected: {r.expected} | Got: {r.actual ?? 'error'}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Feedback */}
              {tutorFeedback && (
                <div style={{
                  padding: '12px 16px', borderRadius: '10px', marginBottom: '12px',
                  backgroundColor: tutorFeedback.correct ? 'rgba(0,204,136,0.15)' : 'rgba(255,170,50,0.15)',
                  border: `1px solid ${tutorFeedback.correct ? '#00cc8844' : '#ffaa3344'}`,
                }}>
                  <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '6px', color: tutorFeedback.correct ? '#00cc88' : '#ffaa33' }}>
                    {tutorFeedback.correct ? '✓ Correct!' : '✗ Not quite right'}
                  </div>
                  <div style={{ fontSize: '13px', lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>{tutorFeedback.feedback}</div>
                  {tutorFeedback.missing_points && tutorFeedback.missing_points.length > 0 && (
                    <div style={{ marginTop: '8px', fontSize: '12px', color: '#ffaa33' }}>
                      Missing: {tutorFeedback.missing_points.join(', ')}
                    </div>
                  )}
                  <div style={{ marginTop: '6px', fontSize: '11px', color: '#888' }}>Attempts: {tutorFeedback.attempts}</div>
                </div>
              )}

              {/* Hint */}
              {tutorHint && (
                <div style={{ padding: '10px 14px', borderRadius: '8px', backgroundColor: 'rgba(85,51,255,0.15)', border: '1px solid rgba(85,51,255,0.3)', marginBottom: '12px', fontSize: '13px' }}>
                  💡 {tutorHint}
                </div>
              )}

              {/* Hint / Next buttons */}
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={handleTutorHint} disabled={tutorLoading || (tutorFeedback?.solved ?? false)}
                  style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid #5533ff', backgroundColor: 'transparent', color: '#5533ff', cursor: 'pointer', fontSize: '12px' }}>
                  💡 Hint
                </button>
                {tutorFeedback?.solved && (
                  <button onClick={handleTutorGenerate}
                    style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', backgroundColor: '#00cc88', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '12px' }}>
                    Next Problem →
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Empty state */}
          {!tutorProblem && !tutorLoading && (
            <div style={{ textAlign: 'center', padding: '40px 0', color: '#666', fontSize: '14px' }}>
              Choose a topic and click "Generate Problem" to start learning.
            </div>
          )}
          {tutorLoading && !tutorProblem && (
            <div style={{ textAlign: 'center', padding: '40px 0', color: '#00cc88', fontSize: '14px' }}>
              Generating problem...
            </div>
          )}
        </div>
      )}

      {/* Chat history — infinite scroll */}
      {activeMode !== 'tutor' && (
        <div
        ref={chatContainerRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          // Load more when scrolled near the top
          if (el.scrollTop < 60 && visibleCount < messages.length) {
            setVisibleCount(prev => Math.min(prev + MESSAGES_PER_PAGE, messages.length));
          }
        }}
        style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 30,
            width: '80%',
            maxWidth: '960px',
            maxHeight: '70vh',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            padding: '16px',
            borderRadius: '20px',  // Softer
            // Glassmorphism for the whole box:
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            backdropFilter: 'blur(25px)',
            WebkitBackdropFilter: 'blur(25px)',
            border: '1px solid rgba(255, 255, 255, 0.18)',
            boxShadow: '0 12px 40px rgba(0, 0, 0, 0.5)',
        }}
        >
        {/* Load more button when there are older messages */}
        {messages.length > visibleCount && (
          <button
            onClick={() => setVisibleCount(prev => Math.min(prev + MESSAGES_PER_PAGE, messages.length))}
            style={{
              alignSelf: 'center', padding: '6px 16px', borderRadius: '12px',
              border: '1px solid rgba(85,51,255,0.3)', backgroundColor: 'rgba(85,51,255,0.1)',
              color: '#aaa', cursor: 'pointer', fontSize: '11px', marginBottom: '8px',
            }}
          >
            Load earlier messages ({messages.length - visibleCount} more)
          </button>
        )}
        {messages.slice(-visibleCount).map((m, sliceIdx) => {
            const i = Math.max(0, messages.length - visibleCount) + sliceIdx; // real index
            const isExplanation = m.role === 'ai' && m.text.startsWith('[EXPLANATION]');
            return (
            <div
            key={i}
            style={{
                padding: '12px 16px',
                backgroundColor: m.role === 'user'
                  ? 'rgba(128, 0, 128, 0.5)'
                  : isExplanation
                    ? 'rgba(80, 140, 255, 0.12)'
                    : 'rgba(255, 255, 255, 0.12)',
                backdropFilter: 'blur(15px)',
                WebkitBackdropFilter: 'blur(15px)',
                borderRadius: '18px',
                border: isExplanation
                  ? '1px solid rgba(100, 160, 255, 0.3)'
                  : '1px solid rgba(255, 255, 255, 0.1)',
                borderLeft: isExplanation ? '3px solid rgba(100, 160, 255, 0.6)' : undefined,
                color: 'rgba(255, 255, 255, 0.95)',
                textShadow: '0 1px 2px rgba(0,0,0,0.5)', 
                maxWidth: '85%', 
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            }}
            >
            <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, wordBreak: 'break-word', fontFamily: m.isDiff ? 'monospace' : 'inherit' }}>
              {m.imageData && (
                <div style={{ marginBottom: '8px' }}>
                  <img
                    src={m.imageData}
                    alt="screenshot"
                    onClick={() => setEnlargedImage(m.imageData!)}
                    style={{
                      maxWidth: '220px', maxHeight: '140px',
                      borderRadius: '8px', border: '1px solid rgba(255,255,255,0.2)',
                      cursor: 'zoom-in', display: 'block', objectFit: 'cover',
                    }}
                    title="Click to enlarge"
                  />
                </div>
              )}
              {m.isDiff
                ? m.text.split('\n').map((line, li) => (
                    <span key={li} style={{
                      display: 'block',
                      backgroundColor: line.startsWith('+') && !line.startsWith('+++') ? 'rgba(0,200,80,0.18)'
                                     : line.startsWith('-') && !line.startsWith('---') ? 'rgba(220,50,50,0.18)'
                                     : line.startsWith('@@') ? 'rgba(100,150,255,0.15)'
                                     : 'transparent',
                      color: line.startsWith('+') && !line.startsWith('+++') ? '#7fffA0'
                           : line.startsWith('-') && !line.startsWith('---') ? '#ff8080'
                           : line.startsWith('@@') ? '#aac4ff'
                           : 'inherit',
                    }}>{line || '\u00a0'}</span>
                  ))
                : m.text
              }
            </div>
            {m.role === 'ai' && i === messages.length - 1 && lastUserQuery && activeMode === 'query' && queryMode !== 'deep_semantic' && (
              <div style={{ marginTop: 6 }}>
                <button
                  onClick={async () => {
                    if (!lastUserQuery) return;
                    setMode('querying');
                    try {
                      const res = await fetch('http://localhost:8000/feedback', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: lastUserQuery, prev_mode: queryMode, session_id: sessionId })
                      });
                      const deep = await res.json();
                      // append deep answer and summary
                      setMessages(prev => [
                        ...prev,
                        { role: 'ai', text: `[DEEP RETRY] ${deep.answer}` },
                        { role: 'ai', text: `[SUMMARY] ${deep.summary}` }
                      ]);
                      // stash detailed for Show Details
                      setDetailsContent(`CITATIONS:\n${deep.citations}\n\nDETAILED:\n${deep.detailed}`);
                      setLastUserQuery(null);
                    } catch (e) {
                      setMessages(prev => [...prev, { role: 'ai', text: 'Feedback failed.' }]);
                    }
                    setMode('idle');
                  }}
                  style={{
                    marginLeft: 8,
                    padding: '6px 8px',
                    borderRadius: 8,
                    backgroundColor: '#aa2222',
                    color: '#fff',
                    border: 'none',
                    cursor: 'pointer'
                  }}
                >
                  Dislike
                </button>
              </div>
            )}
            {pendingAgentEdit && i === messages.length - 1 && activeMode === 'agent' && (
              <div style={{ marginTop: 12 }}>
                {/* Show context file diffs if any */}
                {pendingAgentEdit.contextEdits && pendingAgentEdit.contextEdits.length > 0 && (
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: '12px', color: '#aaa', marginBottom: 4 }}>Context file changes:</div>
                    {pendingAgentEdit.contextEdits.map((ce, idx) => (
                      <details key={idx} style={{ marginBottom: 4 }}>
                        <summary style={{ fontSize: '12px', color: '#7ec8e3', cursor: 'pointer' }}>
                          {ce.path.split(/[/\\]/).pop()} — {ce.explanation || 'modified'}
                        </summary>
                        <pre style={{ fontSize: '11px', background: '#1a1a2e', padding: 6, borderRadius: 4, overflowX: 'auto', maxHeight: 200 }}>{ce.diff}</pre>
                      </details>
                    ))}
                  </div>
                )}
                <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={async () => {
                    if (!pendingAgentEdit) return;
                    setMode('agent-processing');
                    try {
                      const res = await fetch('http://localhost:8000/agent/apply', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                          file_path: pendingAgentEdit.filePath,
                          instruction: pendingAgentEdit.instruction,
                          confirmed: true,
                          task_mode: agentTaskMode,
                          session_id: sessionId,
                          new_source: pendingAgentEdit.newSource || undefined,
                          context_file_edits: pendingAgentEdit.contextEdits || undefined,
                        })
                      });
                      const result = await res.json();
                      const ctxNote = result.applied_context_files?.length
                        ? ` + ${result.applied_context_files.length} context file(s)`
                        : '';
                      setMessages(prev => [...prev, { role: 'ai', text: (result.message || 'Changes applied successfully!') + ctxNote }]);
                      setPendingAgentEdit(null);
                    } catch (e) {
                      setMessages(prev => [...prev, { role: 'ai', text: 'Error applying changes.' }]);
                    }
                    setMode('idle');
                  }}
                  style={{
                    padding: '6px 12px',
                    borderRadius: 8,
                    backgroundColor: '#22aa22',
                    color: '#fff',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  ✓ Apply Changes
                </button>
                <button
                  onClick={() => {
                    setMessages(prev => [...prev, { role: 'ai', text: 'Changes declined.' }]);
                    setPendingAgentEdit(null);
                  }}
                  style={{
                    padding: '6px 12px',
                    borderRadius: 8,
                    backgroundColor: '#aa2222',
                    color: '#fff',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  ✗ Decline
                </button>
                <button
                  onClick={() => {
                    if (pendingAgentEdit) {
                      setRagChunkPrompt({
                        show: true,
                        instruction: pendingAgentEdit.instruction,
                        filePath: pendingAgentEdit.filePath
                      });
                    }
                  }}
                  style={{
                    padding: '6px 12px',
                    borderRadius: 8,
                    backgroundColor: '#5533ff',
                    color: '#fff',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  🔄 Try with More Context
                </button>
                </div>
              </div>
            )}
            </div>
            );
        })}
        <div ref={messagesEndRef} />
      </div>
      )}

      {/* RAG Chunk Selection Modal */}
      {ragChunkPrompt && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
          }}
          onClick={() => setRagChunkPrompt(null)}
        >
          <div
            style={{
              backgroundColor: '#1a1a1a',
              border: '1px solid #444',
              borderRadius: '12px',
              padding: '24px',
              maxWidth: '400px',
              color: '#fff',
              boxShadow: '0 10px 40px rgba(0, 0, 0, 0.8)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ marginTop: 0, marginBottom: '16px', color: '#5533ff' }}>More Context Needed</h3>
            <p style={{ marginBottom: '12px', lineHeight: '1.5', fontSize: '13px' }}>
              LLM couldn't generate proper code changes. Retry with more context:
            </p>
            
            <div style={{ marginBottom: '14px' }}>
              <p style={{ marginTop: 0, marginBottom: '8px', fontSize: '12px', color: '#aaa' }}>
                Search Method:
              </p>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  onClick={() => setRagSearchMethod('bm25')}
                  style={{
                    flex: 1,
                    padding: '8px',
                    borderRadius: '6px',
                    border: ragSearchMethod === 'bm25' ? '2px solid #22bbff' : '1px solid #555',
                    backgroundColor: ragSearchMethod === 'bm25' ? 'rgba(34,187,255,0.2)' : '#1a1a1a',
                    color: ragSearchMethod === 'bm25' ? '#22bbff' : '#aaa',
                    cursor: 'pointer',
                    fontSize: '12px',
                    fontWeight: ragSearchMethod === 'bm25' ? 'bold' : 'normal',
                  }}
                >
                  BM25
                </button>
                <button
                  onClick={() => setRagSearchMethod('semantic')}
                  style={{
                    flex: 1,
                    padding: '8px',
                    borderRadius: '6px',
                    border: ragSearchMethod === 'semantic' ? '2px solid #22ff99' : '1px solid #555',
                    backgroundColor: ragSearchMethod === 'semantic' ? 'rgba(34,255,153,0.2)' : '#1a1a1a',
                    color: ragSearchMethod === 'semantic' ? '#22ff99' : '#aaa',
                    cursor: 'pointer',
                    fontSize: '12px',
                    fontWeight: ragSearchMethod === 'semantic' ? 'bold' : 'normal',
                  }}
                >
                  Semantic (🔍)
                </button>
                <button
                  onClick={() => setRagSearchMethod('both')}
                  style={{
                    flex: 1,
                    padding: '8px',
                    borderRadius: '6px',
                    border: ragSearchMethod === 'both' ? '2px solid #ffaa22' : '1px solid #555',
                    backgroundColor: ragSearchMethod === 'both' ? 'rgba(255,170,34,0.2)' : '#1a1a1a',
                    color: ragSearchMethod === 'both' ? '#ffaa22' : '#aaa',
                    cursor: 'pointer',
                    fontSize: '12px',
                    fontWeight: ragSearchMethod === 'both' ? 'bold' : 'normal',
                  }}
                >
                  Both
                </button>
              </div>
            </div>
            
            <p style={{ marginTop: '8px', marginBottom: '14px', fontSize: '11px', color: '#666', lineHeight: '1.4' }}>
              • <strong>BM25</strong>: Keyword search (faster, good for algorithm names)
              <br/>
              • <strong>Semantic</strong>: Vector similarity (catches meaning, ChromaDB)
              <br/>
              • <strong>Both</strong>: BM25 + semantic fallback
            </p>
            
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              <button
                onClick={() => handleRagChunkRetry(5)}
                style={{
                  flex: 1,
                  padding: '10px',
                  borderRadius: '8px',
                  border: 'none',
                  backgroundColor: '#5533ff',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                5 Chunks
              </button>
              <button
                onClick={() => handleRagChunkRetry(7)}
                style={{
                  flex: 1,
                  padding: '10px',
                  borderRadius: '8px',
                  border: 'none',
                  backgroundColor: '#5533ff',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                7 Chunks
              </button>
              <button
                onClick={() => handleRagChunkRetry(10)}
                style={{
                  flex: 1,
                  padding: '10px',
                  borderRadius: '8px',
                  border: 'none',
                  backgroundColor: '#5533ff',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                10 Chunks
              </button>
            </div>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              <input
                type="number"
                min="1"
                max="20"
                value={customChunkInput}
                onChange={(e) => setCustomChunkInput(e.target.value)}
                placeholder="Custom (1-20)"
                style={{
                  flex: 1,
                  padding: '10px',
                  borderRadius: '8px',
                  border: '1px solid #444',
                  backgroundColor: '#111',
                  color: '#fff',
                  fontSize: '14px',
                  outline: 'none',
                }}
              />
              <button
                onClick={() => {
                  const chunks = parseInt(customChunkInput);
                  if (chunks && chunks > 0 && chunks <= 50) {
                    handleRagChunkRetry(chunks);
                  }
                }}
                style={{
                  padding: '10px 16px',
                  borderRadius: '8px',
                  border: 'none',
                  backgroundColor: '#5533ff',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                Go
              </button>
            </div>

            <button
              onClick={() => setRagChunkPrompt(null)}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid #444',
                backgroundColor: '#1a1a1a',
                color: '#999',
                cursor: 'pointer',
                fontSize: '14px',
              }}
            >
              Skip
            </button>
          </div>
        </div>
      )}

      {/* Enlarged image modal */}
      {enlargedImage && (
        <div
          onClick={() => setEnlargedImage(null)}
          style={{
            position: 'fixed', inset: 0, zIndex: 200,
            backgroundColor: 'rgba(0,0,0,0.88)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'zoom-out',
          }}
        >
          <img
            src={enlargedImage}
            alt="enlarged screenshot"
            style={{
              maxWidth: '90vw', maxHeight: '90vh',
              borderRadius: '12px', border: '1px solid rgba(255,255,255,0.15)',
              boxShadow: '0 16px 64px rgba(0,0,0,0.8)',
            }}
            onClick={e => e.stopPropagation()}
          />
          <button
            onClick={() => setEnlargedImage(null)}
            style={{
              position: 'fixed', top: '20px', right: '24px',
              background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '50%', width: '36px', height: '36px',
              color: '#fff', fontSize: '18px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >✕</button>
        </div>
      )}

      {/* Mode toggle — always visible */}
      <button
        onClick={() => {
          const modes: ActiveMode[] = ['query', 'agent', 'tutor'];
          const idx = modes.indexOf(activeMode);
          setActiveMode(modes[(idx + 1) % modes.length]);
        }}
        disabled={mode !== 'idle'}
        style={{
          position: 'fixed', bottom: '28px', right: '40px', zIndex: 50,
          padding: '10px 14px', borderRadius: '12px', border: '2px solid',
          borderColor: activeMode === 'agent' ? '#ff00ff' : activeMode === 'tutor' ? '#00cc88' : '#5533ff',
          backgroundColor: activeMode === 'agent' ? 'rgba(255,0,255,0.1)' : activeMode === 'tutor' ? 'rgba(0,204,136,0.1)' : 'rgba(85,51,255,0.1)',
          color: activeMode === 'agent' ? '#ff00ff' : activeMode === 'tutor' ? '#00cc88' : '#5533ff',
          cursor: 'pointer', fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', transition: 'all 0.3s',
          backdropFilter: 'blur(10px)',
        }}
      >
        {activeMode === 'agent' ? '🤖 Agent' : activeMode === 'tutor' ? '📚 Tutor' : '🔍 Query'}
      </button>

      {/* Input bar fixed at bottom center */}
      {activeMode !== 'tutor' && (
      <form
        onSubmit={handleSubmit}
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '80%',
          maxWidth: '740px',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px',
          zIndex: 20,
        }}
      >
        {/* Agent-mode controls row: Mode + Tests above input */}
        {activeMode === 'agent' && (
          <div style={{ display: 'flex', gap: '6px', justifyContent: 'center', alignItems: 'center' }}>
            <select
              value={agentTaskMode}
              onChange={e => setAgentTaskMode(e.target.value as any)}
              disabled={mode !== 'idle'}
              title="Fix: debug existing code  |  Solve/Build: implement from scratch  |  Auto: detect from prompt"
              style={{ padding: '5px 8px', borderRadius: '8px', backgroundColor: '#111', color: '#fff', border: '1px solid #555', fontSize: '11px', cursor: 'pointer' }}
            >
              <option value="auto">⚙ Auto</option>
              <option value="fix">🔧 Fix</option>
              <option value="solve">🏗 Solve/Build</option>
            </select>
            <button
              type="button"
              onClick={() => setShowTestPanel(p => !p)}
              disabled={mode !== 'idle'}
              title="Toggle test case panel"
              style={{
                padding: '5px 10px', borderRadius: '8px', border: '1px solid',
                borderColor: showTestPanel ? '#22bbff' : '#555',
                backgroundColor: showTestPanel ? 'rgba(34,187,255,0.15)' : '#111',
                color: showTestPanel ? '#22bbff' : '#aaa',
                fontSize: '11px', cursor: 'pointer', whiteSpace: 'nowrap'
              }}
            >
              🧪 Tests{testCases.length > 0 ? ` (${testCases.length})` : ''}
            </button>
          </div>
        )}
        {/* Main input row */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
        {detailsContent && (
          <button
            type="button"
            onClick={() => {
              setMessages(prev => [...prev, { role: 'ai', text: detailsContent }]);
              setDetailsContent(null);
            }}
            disabled={mode !== 'idle'}
            style={{
              marginRight: '8px',
              padding: '10px 12px',
              borderRadius: '12px',
              border: 'none',
              backgroundColor: '#444',
              color: '#fff',
              cursor: 'pointer'
            }}
          >
            Show Details
          </button>
        )}
        {/* Multi-Agent toggle — right next to the Agent/Query button */}
        {activeMode === 'agent' && (
          <button
            type="button"
            onClick={() => setUseMultiAgent(p => !p)}
            disabled={mode !== 'idle'}
            title="Multi-agent: Planner → Code Agent → Critic with retries"
            style={{
              padding: '10px 10px', borderRadius: '12px', border: '2px solid',
              borderColor: useMultiAgent ? '#ff00ff' : '#555',
              backgroundColor: useMultiAgent ? 'rgba(255,0,255,0.15)' : '#111',
              color: useMultiAgent ? '#ff00ff' : '#aaa',
              fontSize: '11px', cursor: 'pointer', whiteSpace: 'nowrap',
              transition: 'all 0.3s',
            }}
          >
            🔗{useMultiAgent ? ' ✓' : ''}
          </button>
        )}
        {/* Voice input button */}
        <button
          type="button"
          onClick={() => {
            if (isRecording) {
              stopVoiceRecording();
            } else {
              startVoiceRecording();
            }
          }}
          disabled={mode !== 'idle' || voiceLoading}
          title="Click to start/stop voice recording (Whisper transcription)"
          style={{
            padding: '10px 12px',
            borderRadius: '12px',
            border: '2px solid',
            borderColor: isRecording ? '#ff4444' : voiceLoading ? '#ff9900' : '#444',
            backgroundColor: isRecording ? 'rgba(255,68,68,0.2)' : voiceLoading ? 'rgba(255,153,0,0.15)' : '#111',
            color: isRecording ? '#ff4444' : voiceLoading ? '#ff9900' : '#888',
            cursor: mode !== 'idle' ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            transition: 'all 0.15s',
            animation: isRecording ? 'pulse 0.8s infinite' : 'none',
          }}
        >
          {voiceLoading ? '⏳' : isRecording ? '⏹' : '🎤'}
        </button>

        {/* OCR capture button with dropdown */}
        <div style={{ position: 'relative' }}>
          <input
            id="ocr-image-picker"
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={async (e) => {
              const f = (e.target as HTMLInputElement).files?.[0];
              if (f) await handleOCRCapture(f);
              (document.getElementById('ocr-image-picker') as HTMLInputElement).value = '';
            }}
          />
          <button
            type="button"
            onClick={() => setShowOcrMenu(p => !p)}
            disabled={mode !== 'idle' || ocrLoading}
            title="OCR: capture screenshot or upload image → extract text"
            style={{
              padding: '10px 12px',
              borderRadius: '12px',
              border: '2px solid',
              borderColor: ocrLoading ? '#ff9900' : showOcrMenu ? '#22bbff' : '#444',
              backgroundColor: ocrLoading ? 'rgba(255,153,0,0.15)' : showOcrMenu ? 'rgba(34,187,255,0.15)' : '#111',
              color: ocrLoading ? '#ff9900' : '#888',
              cursor: mode !== 'idle' || ocrLoading ? 'not-allowed' : 'pointer',
              fontSize: '16px',
              transition: 'all 0.15s',
            }}
          >
            {ocrLoading ? '⏳' : '📷'}
          </button>
          {showOcrMenu && (
            <div style={{
              position: 'absolute', bottom: '48px', left: '50%', transform: 'translateX(-50%)',
              backgroundColor: 'rgba(20,20,30,0.97)', border: '1px solid rgba(85,51,255,0.4)',
              borderRadius: '10px', padding: '6px', display: 'flex', flexDirection: 'column', gap: '4px',
              minWidth: '160px', boxShadow: '0 8px 24px rgba(0,0,0,0.6)', zIndex: 100,
            }}>
              <button
                type="button"
                onClick={handleScreenCapture}
                style={{
                  padding: '8px 12px', borderRadius: '6px', border: 'none',
                  backgroundColor: 'transparent', color: '#fff', cursor: 'pointer',
                  fontSize: '13px', textAlign: 'left',
                }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(85,51,255,0.2)')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                🖥 Capture Screen
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowOcrMenu(false);
                  (document.getElementById('ocr-image-picker') as HTMLInputElement)?.click();
                }}
                style={{
                  padding: '8px 12px', borderRadius: '6px', border: 'none',
                  backgroundColor: 'transparent', color: '#fff', cursor: 'pointer',
                  fontSize: '13px', textAlign: 'left',
                }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(85,51,255,0.2)')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                📁 Upload Image
              </button>
            </div>
          )}
        </div>

        {/* Pending image indicator */}
        {pendingImageBlob && (
          <div style={{
            padding: '4px 10px', borderRadius: '12px', backgroundColor: 'rgba(85,51,255,0.2)',
            border: '1px solid rgba(85,51,255,0.4)', fontSize: '11px', color: '#b388ff',
            display: 'flex', alignItems: 'center', gap: '6px', marginRight: '4px',
          }}>
            📷 Image attached
            <button onClick={() => { setPendingImageBlob(null); setPendingImageDataUrl(null); }}
              style={{ background: 'none', border: 'none', color: '#ff6666', cursor: 'pointer', fontSize: '11px', padding: '0 2px' }}>✕</button>
          </div>
        )}

        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={pendingImageBlob ? 'Ask about the captured image...' : activeMode === 'agent' ? 'Describe code changes...' : 'Ask Aion...'}
          disabled={mode !== 'idle'}
          style={{
            flex: 1,
            padding: '12px 18px',
            borderRadius: '999px',
            border: '1px solid #333',
            backgroundColor: '#111',
            color: '#fff',
            fontSize: '14px',
            outline: 'none',
          }}
        />
        <button
          type="submit"
          disabled={mode !== 'idle' || !input.trim()}
          style={{
            padding: '12px 24px',
            borderRadius: '999px',
            border: 'none',
            backgroundColor: mode !== 'idle' ? '#222' : '#5533ff',
            color: '#fff',
            cursor: mode !== 'idle' ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            transition: 'background 0.3s, transform 0.1s',
          }}
        >
          {mode !== 'idle' ? '...' : 'Send'}
        </button>
        </div>
      </form>
      )}
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);

// Inject pulse keyframe for recording animation
const styleEl = document.createElement('style');
styleEl.textContent = `@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(255,68,68,0.5)} 50%{box-shadow:0 0 0 6px rgba(255,68,68,0.0)} }`;
document.head.appendChild(styleEl);

