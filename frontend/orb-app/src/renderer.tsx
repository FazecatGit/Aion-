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
  const [useTwoMode, setUseTwoMode] = useState(false);  // two-mode agent: auto-routes between Do-It and Explain
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
  // Persistent OCR context: keeps the extracted text/analysis available for follow-up queries
  // until the user explicitly clears it (e.g., by capturing a new image or dismissing)
  const [ocrContext, setOcrContext] = useState<{ text: string; imageDataUrl: string } | null>(null);

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
  const [tutorStyle, setTutorStyle] = useState<'mcq' | 'free_text' | 'code' | 'solve' | 'proof'>('mcq');
  const [tutorProblem, setTutorProblem] = useState<TutorProblem | null>(null);
  const [tutorFeedback, setTutorFeedback] = useState<TutorFeedback | null>(null);
  const [tutorHint, setTutorHint] = useState<string | null>(null);
  const [tutorCode, setTutorCode] = useState('');
  const [tutorAnswer, setTutorAnswer] = useState('');
  const [tutorCodeResults, setTutorCodeResults] = useState<{ input: string; expected: string; actual: string | null; passed: boolean }[] | null>(null);
  const [tutorLoading, setTutorLoading] = useState(false);
  const [showLesson, setShowLesson] = useState(true);
  const [agentLearnings, setAgentLearnings] = useState<{ topic: string; explanation: string; timestamp: string }[] | null>(null);
  const [showLearnings, setShowLearnings] = useState(false);

  // ── Gamification state ──────────────────────────────────────────────────────
  type GamifBadge = { id: string; name: string; icon: string; desc: string };
  type GamifProfile = {
    xp: number; level: number;
    level_progress: { level: number; current_xp_in_level: number; xp_needed_for_next: number; progress_pct: number };
    badges: GamifBadge[]; all_badges: GamifBadge[];
    streak_days: number; last_activity_date: string | null;
    total_solved: number; total_attempted: number;
    topic_streaks: Record<string, number>; daily_xp: Record<string, number>;
  };
  const [gamifProfile, setGamifProfile] = useState<GamifProfile | null>(null);
  const [showGamifPanel, setShowGamifPanel] = useState(false);

  // ── Problem bank state ─────────────────────────────────────────────────────
  type ProblemBankStats = { total_problems: number; by_category: Record<string, number>; by_difficulty: Record<string, number>; categories: string[] };
  const [showProblemBank, setShowProblemBank] = useState(false);
  const [problemBankStats, setProblemBankStats] = useState<ProblemBankStats | null>(null);
  const [pbCategory, setPbCategory] = useState('general');
  const [pbDifficulty, setPbDifficulty] = useState('medium');

  const fetchGamifProfile = async () => {
    try {
      const res = await fetch('http://localhost:8000/gamification/profile');
      const data = await res.json();
      if (data.status === 'ok') {
        const { status, ...profile } = data;
        setGamifProfile(profile as GamifProfile);
      }
    } catch { /* server not running */ }
  };

  // ── Math mode state ────────────────────────────────────────────────────────
  type GraphPoint = { x: number; y: number };
  const [isMathMode, setIsMathMode] = useState(false);
  const [mathExpression, setMathExpression] = useState('x^2');
  const [mathExpression2, setMathExpression2] = useState('');  // g(x)
  const [mathGraphPoints, setMathGraphPoints] = useState<GraphPoint[]>([]);
  const [mathGraph2Points, setMathGraph2Points] = useState<GraphPoint[]>([]);  // g(x) points
  const [mathDerivPoints, setMathDerivPoints] = useState<GraphPoint[]>([]);
  const [mathDerivExpr, setMathDerivExpr] = useState('');
  const [mathHoverX, setMathHoverX] = useState<number | null>(null);
  const [mathSteps, setMathSteps] = useState<string[] | null>(null);
  const [showMathGraph, setShowMathGraph] = useState(false);
  const tutorAbortRef = useRef<AbortController | null>(null);  // for stop button
  const globalAbortRef = useRef<AbortController | null>(null);  // cancels any in-flight fetch
  const [pulsingBtn, setPulsingBtn] = useState<string | null>(null);  // tracks which button is pulsing

  // ── Math Visualization Tools ───────────────────────────────────────────────
  const [showMathViz, setShowMathViz] = useState(false);
  type MathVizType = 'vector' | 'circle' | 'triangle' | 'unitcircle' | 'matrix' | 'normal' | 'bezier' | null;
  const [mathVizType, setMathVizType] = useState<MathVizType>(null);
  // Vector: {x, y} endpoints
  const [vizVectors, setVizVectors] = useState<{x: number; y: number; label: string; color: string}[]>([
    { x: 3, y: 2, label: 'a', color: '#00cc88' }, { x: -1, y: 4, label: 'b', color: '#22aaff' }
  ]);
  // Circle: center + radius
  const [vizCircle, setVizCircle] = useState({ cx: 0, cy: 0, r: 5 });
  // Triangle: three vertices
  const [vizTriangle, setVizTriangle] = useState<{x: number; y: number}[]>([
    { x: 0, y: 0 }, { x: 6, y: 0 }, { x: 3, y: 5 }
  ]);
  // Unit circle angle in degrees
  const [vizAngle, setVizAngle] = useState(45);

  // ── Curriculum browser state ───────────────────────────────────────────────
  type CurriculumChapter = { id: string; name: string; topics: string[]; completed: boolean };
  type CurriculumSubject = { name: string; icon: string; progress: string; chapters: CurriculumChapter[] };
  const [showCurriculum, setShowCurriculum] = useState(false);
  const [curriculum, setCurriculum] = useState<Record<string, CurriculumSubject> | null>(null);
  const [csCurriculum, setCsCurriculum] = useState<Record<string, CurriculumSubject> | null>(null);
  const [currTab, setCurrTab] = useState<'math' | 'cs'>('math');
  const [currExpandedSubject, setCurrExpandedSubject] = useState<string | null>(null);
  const [currExpandedChapter, setCurrExpandedChapter] = useState<string | null>(null);

  // ── Matrix viz state ───────────────────────────────────────────────────────
  const [vizMatrix, setVizMatrix] = useState<number[][]>([[1, 0], [0, 1]]);
  const [vizMatrixPoints, setVizMatrixPoints] = useState<{x: number; y: number}[]>([
    { x: 1, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }
  ]);

  // ── Normal distribution viz state ──────────────────────────────────────────
  const [vizNormMean, setVizNormMean] = useState(0);
  const [vizNormStd, setVizNormStd] = useState(1);
  const [vizNormSigmaShow, setVizNormSigmaShow] = useState(1); // which σ region to highlight: 1, 2, or 3
  const [vizNormShowArea, setVizNormShowArea] = useState(true);
  const [vizNormCompare, setVizNormCompare] = useState(false); // overlay a second curve
  const [vizNormMean2, setVizNormMean2] = useState(1);
  const [vizNormStd2, setVizNormStd2] = useState(0.5);

  // ── Bezier curve viz state ─────────────────────────────────────────────────
  const [vizBezierPts, setVizBezierPts] = useState<{x: number; y: number}[]>([
    { x: 50, y: 250 }, { x: 150, y: 50 }, { x: 250, y: 50 }, { x: 350, y: 250 }
  ]);
  const [vizBezierT, setVizBezierT] = useState(0.5);
  const [showToolsPanel, setShowToolsPanel] = useState(false);
  const [toolsOutput, setToolsOutput] = useState<string | null>(null);
  const [toolsLoading, setToolsLoading] = useState(false);

  // ── Process / backend log panel ────────────────────────────────────────────
  type ProcessLog = { ts: number; level: string; logger: string; message: string };
  const [showProcessPanel, setShowProcessPanel] = useState(false);
  const [processExpanded, setProcessExpanded] = useState(false);
  const [processLogs, setProcessLogs] = useState<ProcessLog[]>([]);
  const processEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Connect to SSE log stream
  useEffect(() => {
    const es = new EventSource('http://localhost:8000/process/logs');
    eventSourceRef.current = es;
    es.onmessage = (ev) => {
      try {
        const entry: ProcessLog = JSON.parse(ev.data);
        setProcessLogs(prev => {
          const next = [...prev, entry];
          return next.length > 500 ? next.slice(-500) : next;
        });
      } catch { /* ignore malformed */ }
    };
    es.onerror = () => {
      // Reconnect handled automatically by EventSource
    };
    return () => { es.close(); };
  }, []);

  // Auto-scroll process log (only when panel is expanded to avoid constant re-renders during active logging)
  useEffect(() => {
    if (showProcessPanel && showProcessPanel === true && processEndRef.current) {
      // Use requestAnimationFrame to batch scroll updates and avoid janky re-renders
      requestAnimationFrame(() => {
        processEndRef.current?.scrollIntoView({ behavior: 'auto' });  // auto instead of smooth to reduce CPU
      });
    }
  }, [processLogs, showProcessPanel]);

  // ── LLM connection status ──────────────────────────────────────────────────
  const [llmConnected, setLlmConnected] = useState<boolean | null>(null);  // null = unknown
  const [llmModel, setLlmModel] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [modelSwitching, setModelSwitching] = useState(false);

  const fetchModels = async () => {
    try {
      const res = await fetch('http://localhost:8000/models');
      if (res.ok) {
        const data = await res.json();
        setAvailableModels((data.available ?? []).map((m: any) => m.name as string));
        setLlmModel(data.current_model ?? '');
      }
    } catch { /* server not ready */ }
  };

  const handleSwitchModel = async (model: string) => {
    if (model === llmModel || modelSwitching) return;
    setModelSwitching(true);
    setShowModelPicker(false);
    try {
      await fetch('http://localhost:8000/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
      });
      setLlmModel(model);
    } catch { /* ignore */ }
    setModelSwitching(false);
  };

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('http://localhost:8000/health');
        if (res.ok) {
          const data = await res.json();
          setLlmConnected(data.llm_connected ?? false);
          setLlmModel(data.llm_model ?? '');
        } else {
          setLlmConnected(false);
        }
      } catch {
        setLlmConnected(false);
      }
    };
    checkHealth();
    fetchModels();
    const interval = setInterval(checkHealth, 15000);  // poll every 15s
    return () => clearInterval(interval);
  }, []);

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

  // Fetch gamification profile when entering tutor mode
  useEffect(() => {
    if (activeMode === 'tutor') fetchGamifProfile();
  }, [activeMode]);

  // Refresh gamification profile after solving a problem
  useEffect(() => {
    if (tutorFeedback?.solved) fetchGamifProfile();
  }, [tutorFeedback?.solved]);

  // Close model picker on outside click
  useEffect(() => {
    if (!showModelPicker) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Element;
      if (!target.closest('[data-model-picker]')) setShowModelPicker(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showModelPicker]);

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
      // Image will be shown on the user's question message when they submit
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'ai', text: `Image attach error: ${err.message}` }]);
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
      // Image will be shown on the user's question message when they submit
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'ai', text: `Screen capture error: ${err.message}` }]);
    } finally {
      setOcrLoading(false);
    }
  };

  // ── Tutor handlers ─────────────────────────────────────────────────────────
  const handleFetchLearnings = async (overrideTopic?: string, overrideLanguage?: string) => {
    try {
      const params = new URLSearchParams();
      const topic = overrideTopic ?? tutorTopic;
      const language = overrideLanguage ?? tutorLanguage;
      if (topic.trim()) params.set('topic', topic.trim());
      if (language) params.set('language', language);
      params.set('limit', '20');
      const res = await fetch(`http://localhost:8000/tutor/learnings?${params}`);
      const data = await res.json();
      setAgentLearnings(data.learnings || []);
      setShowLearnings(true);
    } catch (e) {
      console.error('Failed to fetch learnings:', e);
      setAgentLearnings([]);
      setShowLearnings(true);
    }
  };

  const handleTutorGenerate = async () => {
    setTutorLoading(true);
    setTutorProblem(null); setTutorFeedback(null); setTutorHint(null);
    setTutorCode(''); setTutorAnswer(''); setTutorCodeResults(null);
    setMathSteps(null);
    try {
      const endpoint = isMathMode ? 'http://localhost:8000/math/start' : 'http://localhost:8000/tutor/start';
      const payload = isMathMode
        ? { topic: tutorTopic, difficulty: tutorDifficulty, style: tutorStyle === 'mcq' ? 'mcq' : tutorStyle === 'proof' ? 'proof' : 'solve' }
        : { topic: tutorTopic, difficulty: tutorDifficulty, language: tutorLanguage, style: tutorStyle };
      const controller = new AbortController();
      tutorAbortRef.current = controller;
      const timeout = setTimeout(() => controller.abort(), 120_000);
      const res = await fetch(endpoint, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload), signal: controller.signal,
      });
      clearTimeout(timeout);
      const data = await res.json();
      if (data.status === 'ok') {
        // Extract only expected TutorProblem fields to avoid leaking raw JSON/extra fields
        const problem: TutorProblem = {
          session_id: data.session_id || '',
          style: data.style || tutorStyle,
          question: data.question || '',
          language: data.language || tutorLanguage,
          options: data.options,
          test_cases: data.test_cases,
          function_name: data.function_name,
          code_snippet: data.code_snippet,
          lesson: data.lesson,
        };
        // Safeguard: if lesson.explanation is raw JSON, try to parse it
        if (problem.lesson && typeof problem.lesson.explanation === 'string') {
          const exp = problem.lesson.explanation.trim();
          if (exp.startsWith('{') && exp.endsWith('}')) {
            try {
              const parsed = JSON.parse(exp);
              if (parsed.explanation) problem.lesson = parsed;
            } catch { /* keep original */ }
          }
        }
        // Guard: if question looks like raw JSON (LLM parse failure), show error
        const q = problem.question.trim();
        if (q.startsWith('{') && q.endsWith('}')) {
          setMessages(prev => [...prev, { role: 'ai', text: '[TUTOR] Problem generation returned malformed data. Retrying in 1 second...' }]);
          // Yield to event loop so UI can update and display logs
          await new Promise(r => setTimeout(r, 1000));
          
          // Auto-retry once with proper timeout and abort control
          const retryController = new AbortController();
          const retryTimeout = setTimeout(() => retryController.abort(), 60_000);
          try {
            const retry = await fetch(endpoint, {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
              signal: retryController.signal,
            });
            clearTimeout(retryTimeout);
            const retryData = await retry.json();
            if (retryData.status === 'ok' && retryData.question && !retryData.question.trim().startsWith('{')) {
              const retryProblem = {
                session_id: retryData.session_id || '',
                style: retryData.style || tutorStyle,
                question: retryData.question || '',
                language: retryData.language || tutorLanguage,
                options: retryData.options,
                test_cases: retryData.test_cases,
                function_name: retryData.function_name,
                code_snippet: retryData.code_snippet,
                lesson: retryData.lesson,
              };
              // Safeguard: if lesson.explanation is raw JSON, try to parse it
              if (retryProblem.lesson && typeof retryProblem.lesson.explanation === 'string') {
                const exp = retryProblem.lesson.explanation.trim();
                if (exp.startsWith('{') && exp.endsWith('}')) {
                  try {
                    const parsed = JSON.parse(exp);
                    if (parsed.explanation) retryProblem.lesson = parsed;
                  } catch { /* keep original */ }
                }
              }
              setTutorProblem(retryProblem);
              setMessages(prev => [...prev, { role: 'ai', text: '[TUTOR] ✓ Retry succeeded!' }]);
            } else {
              setMessages(prev => [...prev, { role: 'ai', text: '[TUTOR] Retry also returned malformed data. Please try again.' }]);
            }
          } catch (retryErr: any) {
            clearTimeout(retryTimeout);
            if (retryErr?.name === 'AbortError') {
              setMessages(prev => [...prev, { role: 'ai', text: '[TUTOR] Retry timeout — request cancelled.' }]);
            } else {
              setMessages(prev => [...prev, { role: 'ai', text: `[TUTOR] Retry failed: ${retryErr?.message || 'unknown'}` }]);
            }
          }
        } else {
          setTutorProblem(problem);
        }
      } else {
        setMessages(prev => [...prev, { role: 'ai', text: `[TUTOR] Error: ${data.error || 'unknown'}` }]);
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        setMessages(prev => [...prev, { role: 'ai', text: '[TUTOR] Request stopped.' }]);
      } else {
        setMessages(prev => [...prev, { role: 'ai', text: '[TUTOR] Failed to connect to server.' }]);
      }
    }
    tutorAbortRef.current = null;
    setTutorLoading(false);
  };

  // Stop ALL running processes — tutor, agent, query, everything
  const handleStopAll = () => {
    // Abort tutor-specific controller
    if (tutorAbortRef.current) {
      tutorAbortRef.current.abort();
      tutorAbortRef.current = null;
    }
    // Abort any global in-flight fetch (agent, query, etc.)
    if (globalAbortRef.current) {
      globalAbortRef.current.abort();
      globalAbortRef.current = null;
    }
    // Tell the backend to cancel any running LLM tasks
    fetch('http://localhost:8000/agent/cancel', { method: 'POST' }).catch(() => {});
    // Reset all loading/processing states
    setTutorLoading(false);
    setMode('idle');
    setOcrLoading(false);
    setToolsLoading(false);
    setMessages(prev => [...prev, { role: 'ai', text: '⏹ All processes stopped.' }]);
  };

  // Pulse animation helper for buttons
  const triggerPulse = (btnId: string) => {
    setPulsingBtn(btnId);
    setTimeout(() => setPulsingBtn(null), 600);
  };

  // Evaluate math expression for graphing (f(x) and optionally g(x))
  const handleMathGraph = async (expr?: string) => {
    const expression = expr || mathExpression;
    if (!expression.trim()) return;
    try {
      const values = Array.from({ length: 201 }, (_, i) => -10 + i * 0.1);
      const res = await fetch('http://localhost:8000/math/evaluate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression, variable: 'x', values }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setMathGraphPoints(data.points || []);
        setMathDerivPoints(data.derivative_points || []);
        setMathDerivExpr(data.derivative_expression || '');
        setShowMathGraph(true);
      }
      // Also evaluate g(x) if provided
      if (mathExpression2.trim()) {
        const res2 = await fetch('http://localhost:8000/math/evaluate', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ expression: mathExpression2, variable: 'x', values }),
        });
        const data2 = await res2.json();
        if (data2.status === 'ok') {
          setMathGraph2Points(data2.points || []);
        }
      } else {
        setMathGraph2Points([]);
      }
    } catch { /* ignore graph errors */ }
  };

  // Fetch step-by-step solution
  const handleMathSteps = async () => {
    if (!tutorProblem) return;
    try {
      const res = await fetch(`http://localhost:8000/math/steps/${tutorProblem.session_id}`);
      const data = await res.json();
      if (data.status === 'ok' && data.steps) {
        setMathSteps(data.steps);
      }
    } catch { /* ignore */ }
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

/** Safely parse a fetch response as JSON, handling non-JSON error responses. */
const safeJson = async (res: Response): Promise<any> => {
  if (!res.ok) {
    // Try to parse JSON error body, fall back to status text
    const text = await res.text();
    try { return JSON.parse(text); } catch { /* not JSON */ }
    throw new Error(`Server error ${res.status}: ${text.slice(0, 200) || res.statusText}`);
  }
  return res.json();
};

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!input.trim()) return;

  // Create a fresh AbortController for this request
  const controller = new AbortController();
  globalAbortRef.current = controller;
  const signal = controller.signal;

  const userMsg = input.trim();
  setInput('');
  // Attach imageData to the question message so the user sees the image is part of their query
  const questionImageData = pendingImageDataUrl || ocrContext?.imageDataUrl || undefined;
  setMessages(prev => [...prev, { role: 'user', text: userMsg, imageData: questionImageData }]);

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
      const res = await fetch('http://localhost:8000/ocr/analyze', { method: 'POST', body: form, signal });
      const data = await safeJson(res);
      if (data.status === 'ok') {
        // Update persistent OCR context so follow-up queries also have it
        setOcrContext({ text: data.image_context || data.ocr_text || data.analysis, imageDataUrl: imageDataUrl || '' });
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
      const endpoint = useTwoMode ? '/agent/two-mode' : (useMultiAgent ? '/agent/orchestrate' : '/agent/edit');
      const label = useTwoMode ? '[TWO-MODE] Auto-routing: Do-It vs Explain...' : (useMultiAgent ? '[MULTI-AGENT] Planner → Agent → Critic pipeline running...' : '[CODE AGENT] Processing your code request...');
      setMessages(prev => [...prev, { role: 'ai', text: label }]);
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,
        body: JSON.stringify({
          instruction: userMsg,
          file_path: selectedFilePath,
          ...(useTwoMode ? {
            source_code: '',
            test_cases: testCases.length > 0 ? testCases.map(tc => ({ input: tc.input, expected: tc.expected })) : [],
          } : {
            task_mode: agentTaskMode,
            session_id: sid,
            context_files: contextFiles,
            ...(useMultiAgent && testCases.length > 0 ? {
              test_cases: testCases.map(tc => ({ input: tc.input, expected: tc.expected })),
            } : {}),
          }),
        }),
      });
      const data = await safeJson(res);

      if (useTwoMode && data.status === 'ok') {
        // Two-mode returns {mode, difficulty, ...}
        const twoModeMsgs: Message[] = [];
        const tMode = data.mode === 'do_it' ? 'Do-It' : data.mode === 'explain' ? 'Explain' : 'Explain (fallback)';
        twoModeMsgs.push({ role: 'ai', text: `[TWO-MODE] Route: ${tMode} · Difficulty: ${data.difficulty}` });
        if (data.do_it_attempted && data.do_it_verdict === 'FAIL') {
          twoModeMsgs.push({ role: 'ai', text: `[DO-IT ATTEMPTED] Tests failed — falling back to explanation mode` });
        }
        if (data.diff) twoModeMsgs.push({ role: 'ai', text: `[DRY RUN PREVIEW]\n\n${data.diff}`, isDiff: true });
        if (data.explanation) twoModeMsgs.push({ role: 'ai', text: `[EXPLANATION]\n${data.explanation}` });
        if (data.plan?.steps) {
          const steps = data.plan.steps.map((s: string, i: number) => `  ${i + 1}. ${s}`).join('\n');
          twoModeMsgs.push({ role: 'ai', text: `[PLAN]\n${steps}` });
        }
        setMessages(prev => [...prev, ...twoModeMsgs]);
        if (data.diff && data.file_path) {
          setPendingAgentEdit({ instruction: userMsg, output: data.diff, filePath: data.file_path, newSource: data.new_source, contextEdits: data.context_file_edits || [] });
          setSelectedFilePath(data.file_path);
        }
      } else if (useMultiAgent && (data.status === 'ok' || data.status === 'pending_review')) {
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
        // Show test results if available (execution-based critique)
        if (data.test_results?.length > 0) {
          const passCount = data.test_results.filter((r: any) => r.passed).length;
          const total = data.test_results.length;
          const testLines = data.test_results.map((r: any, i: number) =>
            `  ${r.passed ? '✓' : '✗'} Test ${i + 1}: input=${r.input}  expected=${r.expected}  got=${r.actual || r.error || 'N/A'}`
          ).join('\n');
          orchestrateMsgs.push({ role: 'ai', text: `[TEST RESULTS] ${passCount}/${total} passing\n${testLines}` });
          setTestResults(data.test_results);
        }
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
      // If there's persistent OCR context, prepend it to the question so the LLM has the image content
      const queryQuestion = ocrContext
        ? `[IMAGE CONTEXT]\n${ocrContext.text}\n\n[QUESTION]\n${userMsg}`
        : userMsg;
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,
        body: JSON.stringify({ question: queryQuestion, mode: queryMode, session_id: sid }),
      });
      const data = await safeJson(res);

      // store last user query for potential dislike feedback
      setLastUserQuery(userMsg);

      if (data.status === 'error') {
        setMessages(prev => [...prev, { role: 'ai', text: `Error: ${data.error || data.answer || 'Unknown error'}` }]);
      } else if (data.fast && data.deep) {
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
  } catch (err: any) {
    if (err?.name === 'AbortError') {
      // User cancelled — already handled by handleStopAll
    } else {
      const errMsg = err?.message || String(err);
      setMessages(prev => [...prev, { role: 'ai', text: `Error: ${errMsg}` }]);
    }
  }

  globalAbortRef.current = null;
  setMode('idle');
};

const handleRunTests = async () => {
  if (!selectedFilePath || testCases.length === 0) return;
  const controller = new AbortController();
  globalAbortRef.current = controller;
  setMode('agent-processing');
  try {
    const res = await fetch('http://localhost:8000/agent/run_tests', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        file_path: selectedFilePath,
        test_cases: testCases.map(tc => ({ input: tc.input, expected: tc.expected })),
      }),
    });
    const data = await safeJson(res);
    if (data.status === 'ok') {
      setTestResults(data.results);
    } else {
      setMessages(prev => [...prev, { role: 'ai', text: `[TEST ERROR] ${data.error}` }]);
    }
  } catch (err: any) {
    if (err?.name !== 'AbortError') setMessages(prev => [...prev, { role: 'ai', text: 'Test run failed — is the server running?' }]);
  }
  globalAbortRef.current = null;
  setMode('idle');
};

const handleFixWithTests = async () => {
  if (!selectedFilePath || testCases.length === 0) return;
  const instruction = lastAgentInstruction.current;
  if (!instruction) {
    setMessages(prev => [...prev, { role: 'ai', text: '[TEST] Send the task description first so the agent knows what to implement.' }]);
    return;
  }
  const controller = new AbortController();
  globalAbortRef.current = controller;
  setMode('agent-processing');
  setMessages(prev => [...prev, { role: 'ai', text: `[AUTO-FIX] Running tests and iterating... (up to 3 attempts)` }]);
  try {
    const res = await fetch('http://localhost:8000/agent/fix_with_tests', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        file_path: selectedFilePath,
        instruction,
        test_cases: testCases.map(tc => ({ input: tc.input, expected: tc.expected })),
        max_retries: 3,
        task_mode: agentTaskMode === 'auto' ? 'solve' : agentTaskMode,
        session_id: sessionId,
      }),
    });
    const data = await safeJson(res);
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
  } catch (err: any) {
    if (err?.name !== 'AbortError') setMessages(prev => [...prev, { role: 'ai', text: 'Auto-fix failed — is the server running?' }]);
  }
  globalAbortRef.current = null;
  setMode('idle');
};

const handleRagChunkRetry = async (chunks: number) => {
  if (!ragChunkPrompt) return;
  
  const controller = new AbortController();
  globalAbortRef.current = controller;
  setMode('agent-processing');
  setMessages(prev => [...prev, { role: 'ai', text: `[CODE AGENT] Retrying with ${chunks} RAG chunks (${ragSearchMethod})...` }]);
  
  try {
    const res = await fetch('http://localhost:8000/agent/edit_with_chunks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({ 
        instruction: ragChunkPrompt.instruction, 
        file_path: ragChunkPrompt.filePath,
        max_chunks: chunks,
        task_mode: agentTaskMode,
        search_method: ragSearchMethod
      }),
    });
    const data = await safeJson(res);
    
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
  } catch (err: any) {
    if (err?.name !== 'AbortError') setMessages(prev => [...prev, { role: 'ai', text: 'Error retrying with custom chunks.' }]);
  }
  
  setRagChunkPrompt(null);
  setCustomChunkInput('');
  globalAbortRef.current = null;
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
              const topics: string[] = Array.isArray(json.topics) ? json.topics : Object.keys(json.topics || {});
              setMessages(prev => [...prev, { 
                role: 'ai', 
                text: `✓ Batch ingest complete! Processed all PDFs in data folder. Extracted ${topics.length} topics: ${topics.slice(0, 10).join(', ')}${topics.length > 10 ? '...' : ''}` 
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

        <button
          onClick={async () => {
            setMode('querying');
            setMessages(prev => [...prev, { role: 'user', text: 'Re-ingesting all documents (force rebuild)...' }]);
            try {
              // Start the background reingest
              let res = await fetch('http://localhost:8000/reingest', { method: 'POST' });
              if (res.status === 404 || res.status === 405) {
                setMessages(prev => [...prev, { role: 'ai', text: '(Server needs restart for force-rebuild — falling back to batch ingest)' }]);
                res = await fetch('http://localhost:8000/ingest', { method: 'POST' });
                if (!res.ok) throw new Error(`Server returned ${res.status}`);
                const json = await res.json();
                const topics: string[] = Array.isArray(json.topics) ? json.topics : Object.keys(json.topics || {});
                setMessages(prev => [...prev, { role: 'ai', text: `✓ Ingest complete! ${topics.length} topics extracted.` }]);
                setMode('idle');
                return;
              }
              if (!res.ok) throw new Error(`Server returned ${res.status}`);
              const startJson = await res.json();
              if (startJson.status === 'already_running') {
                setMessages(prev => [...prev, { role: 'ai', text: 'Re-ingest is already running in the background — check back in a minute.' }]);
                setMode('idle');
                return;
              }
              setMessages(prev => [...prev, { role: 'ai', text: ' Starting re-ingest... polling for progress.' }]);

              // Poll /reingest/status with log_cursor for incremental verbose logs
              let done = false;
              let logCursor = 0;
              while (!done) {
                await new Promise(r => setTimeout(r, 3000));
                const statusRes = await fetch(`http://localhost:8000/reingest/status?log_cursor=${logCursor}`);
                const statusJson = await statusRes.json();

                // Show new log lines as individual messages
                const newLines: string[] = statusJson.log || [];
                if (newLines.length > 0) {
                  setMessages(prev => [...prev, ...newLines.map((line: string) => ({ role: 'ai' as const, text: `[ingest] ${line}` }))]);
                  logCursor = statusJson.log_cursor ?? (logCursor + newLines.length);
                }

                if (statusJson.status === 'done') {
                  done = true;
                  const topics: string[] = statusJson.topics || [];
                  setMessages(prev => [...prev, {
                    role: 'ai',
                    text: `Re-ingest complete! ${topics.length} topics extracted: ${topics.slice(0, 15).join(', ')}${topics.length > 15 ? '...' : ''}`
                  }]);
                } else if (statusJson.status === 'error') {
                  done = true;
                  setMessages(prev => [...prev, {
                    role: 'ai',
                    text: `Re-ingest stopped: ${statusJson.error}\n\nProgress has been saved — click Re-ingest again to resume from where it left off.`
                  }]);
                }
              }
            } catch (e: any) {
              setMessages(prev => [...prev, { role: 'ai', text: `Re-ingest failed: ${e.message}\n\nClick Re-ingest again to resume from the last checkpoint.` }]);
            }
            setMode('idle');
          }}
          style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#b85820', color: '#fff', border: 'none', fontWeight: 'bold' }}
          disabled={mode !== 'idle'}
          title="Force full re-ingestion: rebuilds everything with improved keyword extraction and topic tagging"
        >
          🔄 Re-ingest
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
          zIndex: 35, width: '94%', maxWidth: '1400px', maxHeight: '92vh', overflowY: 'auto',
          padding: '28px', borderRadius: '20px',
          backgroundColor: 'rgba(0, 20, 15, 0.94)', backdropFilter: 'blur(25px)',
          border: '1px solid rgba(0,204,136,0.3)', boxShadow: '0 12px 40px rgba(0,0,0,0.6)',
        }}>
          {/* Setup bar */}
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px', alignItems: 'center' }}>
            <input value={tutorTopic} onChange={e => setTutorTopic(e.target.value)} disabled={tutorLoading} placeholder="Topic (e.g. arrays, recursion)"
              style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', flex: '1', minWidth: '120px', opacity: tutorLoading ? 0.5 : 1, cursor: tutorLoading ? 'not-allowed' : 'text' }} />
            <button onClick={async () => {
              const tab = isMathMode ? 'math' : 'cs';
              setCurrTab(tab);
              try {
                if (tab === 'math' && !curriculum) {
                  const r = await fetch('http://localhost:8000/math/curriculum');
                  const d = await r.json();
                  if (d.curriculum) setCurriculum(d.curriculum);
                }
                if (tab === 'cs' && !csCurriculum) {
                  const r = await fetch('http://localhost:8000/cs/curriculum');
                  const d = await r.json();
                  if (d.curriculum) setCsCurriculum(d.curriculum);
                }
              } catch {}
              setShowCurriculum(v => !v);
            }}
              disabled={tutorLoading}
              style={{ padding: '8px 14px', borderRadius: '8px', border: '1px solid #00cc8866', backgroundColor: showCurriculum ? 'rgba(0,204,136,0.15)' : 'transparent', color: '#00cc88', fontWeight: 'bold', cursor: tutorLoading ? 'not-allowed' : 'pointer', fontSize: '12px', opacity: tutorLoading ? 0.5 : 1 }}>
              Browse Topics
            </button>
            <select value={tutorDifficulty} onChange={e => setTutorDifficulty(e.target.value as any)} disabled={tutorLoading}
              style={{ padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', opacity: tutorLoading ? 0.5 : 1, cursor: tutorLoading ? 'not-allowed' : 'pointer' }}>
              <option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option>
            </select>
            {/* Math / CS mode toggle */}
            <button
              type="button"
              onClick={() => setIsMathMode(p => !p)}
              disabled={tutorLoading}
              style={{
                padding: '6px 14px', borderRadius: '8px', fontSize: '12px', fontWeight: 'bold',
                border: '1px solid',
                borderColor: isMathMode ? '#00cc88' : '#5533ff',
                backgroundColor: isMathMode ? 'rgba(0,204,136,0.15)' : 'rgba(85,51,255,0.12)',
                color: isMathMode ? '#00cc88' : '#b388ff',
                cursor: tutorLoading ? 'not-allowed' : 'pointer', transition: 'all 0.2s',
                opacity: tutorLoading ? 0.5 : 1,
              }}
            >
              {isMathMode ? 'Math' : 'CS'}
            </button>
            {/* Language selector - hidden in math mode */}
            {!isMathMode && (
              <select value={tutorLanguage} onChange={e => setTutorLanguage(e.target.value)} disabled={tutorLoading}
                style={{ padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', opacity: tutorLoading ? 0.5 : 1, cursor: tutorLoading ? 'not-allowed' : 'pointer' }}>
                {['python', 'go', 'cpp', 'c', 'javascript', 'typescript', 'java', 'rust'].map(l =>
                  <option key={l} value={l}>{l}</option>
                )}
              </select>
            )}
            <select value={tutorStyle} onChange={e => setTutorStyle(e.target.value as any)} disabled={tutorLoading}
              style={{ padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', opacity: tutorLoading ? 0.5 : 1, cursor: tutorLoading ? 'not-allowed' : 'pointer' }}>
              <option value="mcq">Multiple Choice</option><option value="free_text">Short Answer</option>
              {!isMathMode && <option value="code">Coding</option>}
              {isMathMode && <option value="solve">Solve</option>}
              {isMathMode && <option value="proof">Proof</option>}
            </select>
            <button onClick={() => { triggerPulse('generate'); handleTutorGenerate(); setShowLesson(true); }} disabled={tutorLoading || !tutorTopic.trim()}
              style={{
                padding: '8px 16px', borderRadius: '8px', border: 'none', backgroundColor: '#00cc88', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '13px', opacity: tutorLoading ? 0.5 : 1,
                animation: pulsingBtn === 'generate' ? 'greenPulse 0.6s ease-out' : undefined,
              }}>
              {tutorLoading ? '...' : 'Generate Problem'}
            </button>
            {tutorLoading && (
              <button onClick={handleStopAll}
                style={{ padding: '8px 16px', borderRadius: '8px', border: '2px solid #ff4444', backgroundColor: 'rgba(255,68,68,0.15)', color: '#ff4444', fontWeight: 'bold', cursor: 'pointer', fontSize: '13px' }}>
                Stop
              </button>
            )}
            <button onClick={() => handleFetchLearnings()} disabled={tutorLoading}
              style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid #5533ff', backgroundColor: 'transparent', color: '#b388ff', fontWeight: 'bold', cursor: tutorLoading ? 'not-allowed' : 'pointer', fontSize: '13px', opacity: tutorLoading ? 0.5 : 1 }}>
              Learnings
            </button>
            <button onClick={async () => {
              setShowProblemBank(v => !v);
              if (!problemBankStats) {
                try {
                  const r = await fetch('http://localhost:8000/problem-bank/stats');
                  const d = await r.json();
                  if (d.status === 'ok') { const { status, ...stats } = d; setProblemBankStats(stats as ProblemBankStats); }
                } catch {}
              }
            }} disabled={tutorLoading}
              style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid #ff990066', backgroundColor: showProblemBank ? 'rgba(255,153,0,0.15)' : 'transparent', color: '#ff9900', fontWeight: 'bold', cursor: tutorLoading ? 'not-allowed' : 'pointer', fontSize: '13px', opacity: tutorLoading ? 0.5 : 1 }}>
              Problem Bank
            </button>
          </div>

          {/* ── Gamification Status Bar ─────────────────────────────────────── */}
          {gamifProfile && (
            <div style={{ marginBottom: '12px' }}>
              {/* Compact stats row */}
              <div
                onClick={() => setShowGamifPanel(p => !p)}
                style={{
                  display: 'flex', gap: '12px', alignItems: 'center', padding: '8px 14px', borderRadius: '10px',
                  backgroundColor: 'rgba(255,204,0,0.04)', border: '1px solid rgba(255,204,0,0.2)',
                  cursor: 'pointer', transition: 'background 0.15s', userSelect: 'none',
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.backgroundColor = 'rgba(255,204,0,0.08)'; }}
                onMouseLeave={e => { (e.currentTarget as HTMLElement).style.backgroundColor = 'rgba(255,204,0,0.04)'; }}
              >
                {/* Level badge */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '16px' }}>⚡</span>
                  <span style={{ color: '#ffcc00', fontWeight: 'bold', fontSize: '13px' }}>Lv.{gamifProfile.level}</span>
                </div>
                {/* XP progress bar */}
                <div style={{ flex: 1, maxWidth: '200px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#888', marginBottom: '2px' }}>
                    <span>{gamifProfile.level_progress.current_xp_in_level} / {gamifProfile.level_progress.xp_needed_for_next} XP</span>
                    <span>{gamifProfile.level_progress.progress_pct}%</span>
                  </div>
                  <div style={{ height: '6px', borderRadius: '3px', backgroundColor: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', borderRadius: '3px', transition: 'width 0.5s ease-out',
                      width: `${Math.min(gamifProfile.level_progress.progress_pct, 100)}%`,
                      background: 'linear-gradient(90deg, #ffcc00, #ff9900)',
                    }} />
                  </div>
                </div>
                {/* Total XP */}
                <span style={{ color: '#ffcc00', fontSize: '12px', fontFamily: 'monospace' }}>{gamifProfile.xp} XP</span>
                {/* Streak */}
                {gamifProfile.streak_days > 0 && (
                  <span style={{ color: '#ff6633', fontSize: '12px', fontWeight: 'bold' }}>🔥 {gamifProfile.streak_days}d</span>
                )}
                {/* Solved count */}
                <span style={{ color: '#888', fontSize: '11px' }}>✓ {gamifProfile.total_solved}/{gamifProfile.total_attempted}</span>
                {/* Recent badges (show up to 3) */}
                <div style={{ display: 'flex', gap: '2px' }}>
                  {gamifProfile.badges.slice(-3).map(b => (
                    <span key={b.id} title={`${b.name}: ${b.desc}`} style={{ fontSize: '14px', cursor: 'default' }}>{b.icon}</span>
                  ))}
                </div>
                <span style={{ color: '#555', fontSize: '10px' }}>{showGamifPanel ? '▲' : '▼'}</span>
              </div>

              {/* Expanded gamification panel */}
              {showGamifPanel && (
                <div style={{ marginTop: '8px', padding: '14px', borderRadius: '10px', backgroundColor: 'rgba(255,204,0,0.03)', border: '1px solid rgba(255,204,0,0.15)' }}>
                  {/* Stats grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '8px', marginBottom: '14px' }}>
                    <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(255,204,0,0.08)', textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#ffcc00' }}>{gamifProfile.level}</div>
                      <div style={{ fontSize: '10px', color: '#888' }}>Level</div>
                    </div>
                    <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(0,204,136,0.08)', textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#00cc88' }}>{gamifProfile.xp}</div>
                      <div style={{ fontSize: '10px', color: '#888' }}>Total XP</div>
                    </div>
                    <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(255,102,51,0.08)', textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#ff6633' }}>{gamifProfile.streak_days}</div>
                      <div style={{ fontSize: '10px', color: '#888' }}>Day Streak</div>
                    </div>
                    <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(85,51,255,0.08)', textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#b388ff' }}>{gamifProfile.total_solved}</div>
                      <div style={{ fontSize: '10px', color: '#888' }}>Solved</div>
                    </div>
                  </div>

                  {/* Badge gallery */}
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#ccc', marginBottom: '8px' }}>Badges</div>
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                      {gamifProfile.all_badges.map(b => {
                        const earned = gamifProfile.badges.some(eb => eb.id === b.id);
                        return (
                          <div key={b.id} title={`${b.name}: ${b.desc}`}
                            style={{
                              width: '44px', height: '44px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                              borderRadius: '8px', fontSize: '20px',
                              backgroundColor: earned ? 'rgba(255,204,0,0.12)' : 'rgba(255,255,255,0.03)',
                              border: earned ? '1px solid rgba(255,204,0,0.4)' : '1px solid rgba(255,255,255,0.06)',
                              opacity: earned ? 1 : 0.35, filter: earned ? 'none' : 'grayscale(1)',
                              transition: 'all 0.2s',
                            }}>
                            {b.icon}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Topic streaks */}
                  {Object.keys(gamifProfile.topic_streaks).length > 0 && (
                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#ccc', marginBottom: '6px' }}>Topic Streaks</div>
                      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                        {Object.entries(gamifProfile.topic_streaks)
                          .filter(([, v]) => v > 0)
                          .sort(([, a], [, b]) => b - a)
                          .slice(0, 8)
                          .map(([topic, streak]) => (
                            <span key={topic} style={{
                              padding: '3px 8px', borderRadius: '10px', fontSize: '11px',
                              backgroundColor: streak >= 3 ? 'rgba(255,204,0,0.12)' : 'rgba(255,255,255,0.05)',
                              border: `1px solid ${streak >= 3 ? 'rgba(255,204,0,0.3)' : 'rgba(255,255,255,0.1)'}`,
                              color: streak >= 3 ? '#ffcc00' : '#888',
                            }}>
                              {topic} ×{streak}
                            </span>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Reset button */}
                  <button onClick={async () => {
                    if (!confirm('Reset all gamification progress? This cannot be undone.')) return;
                    try {
                      await fetch('http://localhost:8000/gamification/reset', { method: 'POST' });
                      fetchGamifProfile();
                    } catch {}
                  }}
                    style={{ padding: '4px 12px', borderRadius: '6px', border: '1px solid #ff444444', backgroundColor: 'transparent', color: '#ff4444', cursor: 'pointer', fontSize: '11px' }}>
                    Reset Progress
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ── Problem Bank Browser ────────────────────────────────────────── */}
          {showProblemBank && (
            <div style={{ marginBottom: '16px', padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(255,153,0,0.04)', border: '1px solid rgba(255,153,0,0.2)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#ff9900' }}>📦 Problem Bank</span>
                <button onClick={() => setShowProblemBank(false)} style={{ background: 'none', border: '1px solid #ff990044', borderRadius: '6px', color: '#888', cursor: 'pointer', padding: '4px 10px', fontSize: '11px' }}>Close</button>
              </div>
              {problemBankStats && (
                <div style={{ marginBottom: '12px', fontSize: '12px', color: '#888' }}>
                  {problemBankStats.total_problems} verified problems
                  {problemBankStats.categories.length > 0 && (<span> · Categories: {problemBankStats.categories.join(', ')}</span>)}
                </div>
              )}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                <select value={pbCategory} onChange={e => setPbCategory(e.target.value)}
                  style={{ padding: '6px 10px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }}>
                  <option value="general">General</option>
                  {problemBankStats?.categories.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
                <select value={pbDifficulty} onChange={e => setPbDifficulty(e.target.value)}
                  style={{ padding: '6px 10px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }}>
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
                <button onClick={async () => {
                  setTutorLoading(true);
                  try {
                    const r = await fetch('http://localhost:8000/problem-bank/get', {
                      method: 'POST', headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ category: pbCategory, difficulty: pbDifficulty }),
                    });
                    const d = await r.json();
                    if (d.status === 'ok' && d.problem) {
                      const p = d.problem;
                      setTutorTopic(p.topic);
                      setTutorDifficulty(p.difficulty);
                      setTutorStyle(p.style || 'solve');
                      setTutorProblem({
                        session_id: p.id,
                        style: p.style || 'solve',
                        question: p.question,
                        language: tutorLanguage,
                        options: p.options || [],
                        test_cases: p.test_cases || [],
                      });
                      setTutorFeedback(null); setTutorHint(null); setTutorAnswer(''); setTutorCode('');
                      setShowProblemBank(false);
                    }
                  } catch {}
                  setTutorLoading(false);
                }}
                  disabled={tutorLoading}
                  style={{ padding: '6px 14px', borderRadius: '6px', border: 'none', backgroundColor: '#ff9900', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '12px' }}>
                  Get Problem
                </button>
                <button onClick={async () => {
                  try {
                    await fetch('http://localhost:8000/problem-bank/ingest', { method: 'POST' });
                    const r = await fetch('http://localhost:8000/problem-bank/stats');
                    const d = await r.json();
                    if (d.status === 'ok') { const { status, ...stats } = d; setProblemBankStats(stats as ProblemBankStats); }
                  } catch {}
                }}
                  style={{ padding: '6px 14px', borderRadius: '6px', border: '1px solid #ff990044', backgroundColor: 'transparent', color: '#ff9900', cursor: 'pointer', fontSize: '11px' }}>
                  Ingest from RAG
                </button>
              </div>
            </div>
          )}

          {/* ── Curriculum Browser ──────────────────────────────────────────── */}
          {showCurriculum && (
            <div style={{ marginBottom: '16px', padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(0,204,136,0.04)', border: '1px solid rgba(0,204,136,0.2)', maxHeight: '50vh', overflowY: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                  <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#ccc', marginRight: '4px' }}>Topics</span>
                  <button onClick={async () => {
                    if (!curriculum) {
                      try { const r = await fetch('http://localhost:8000/math/curriculum'); const d = await r.json(); if (d.curriculum) setCurriculum(d.curriculum); } catch {}
                    }
                    setCurrTab('math'); setCurrExpandedSubject(null); setCurrExpandedChapter(null);
                  }} style={{ padding: '4px 12px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold', backgroundColor: currTab === 'math' ? 'rgba(0,204,136,0.25)' : 'rgba(255,255,255,0.05)', color: currTab === 'math' ? '#00cc88' : '#888' }}>Math</button>
                  <button onClick={async () => {
                    if (!csCurriculum) {
                      try { const r = await fetch('http://localhost:8000/cs/curriculum'); const d = await r.json(); if (d.curriculum) setCsCurriculum(d.curriculum); } catch {}
                    }
                    setCurrTab('cs'); setCurrExpandedSubject(null); setCurrExpandedChapter(null);
                  }} style={{ padding: '4px 12px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold', backgroundColor: currTab === 'cs' ? 'rgba(85,51,255,0.25)' : 'rgba(255,255,255,0.05)', color: currTab === 'cs' ? '#b388ff' : '#888' }}>CS</button>
                </div>
                <button onClick={() => setShowCurriculum(false)} style={{ background: 'none', border: '1px solid #00cc8844', borderRadius: '6px', color: '#888', cursor: 'pointer', padding: '4px 10px', fontSize: '11px' }}>Close</button>
              </div>
              {(() => {
                const activeCurr = currTab === 'cs' ? csCurriculum : curriculum;
                const accentColor = currTab === 'cs' ? '#b388ff' : '#00cc88';
                const accentRgb = currTab === 'cs' ? '85,51,255' : '0,204,136';
                if (!activeCurr) return <div style={{ color: '#888', fontSize: '13px' }}>Loading...</div>;
                return (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '12px' }}>
                    {Object.entries(activeCurr).map(([subjId, subj]) => (
                      <div key={subjId} style={{ borderRadius: '10px', border: `1px solid rgba(${accentRgb},0.15)`, backgroundColor: 'rgba(0,0,0,0.25)', overflow: 'hidden' }}>
                        <button onClick={() => setCurrExpandedSubject(currExpandedSubject === subjId ? null : subjId)}
                          style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 14px', background: 'none', border: 'none', color: accentColor, cursor: 'pointer', fontSize: '14px', fontWeight: 'bold', textAlign: 'left' }}>
                          <span>{subj.icon} {subj.name}</span>
                          <span style={{ fontSize: '11px', color: '#888', fontWeight: 'normal' }}>{subj.progress} chapters</span>
                        </button>
                        {currExpandedSubject === subjId && (
                          <div style={{ padding: '0 10px 10px' }}>
                            {subj.chapters.map(ch => (
                              <div key={ch.id} style={{ marginBottom: '4px' }}>
                                <button onClick={() => setCurrExpandedChapter(currExpandedChapter === ch.id ? null : ch.id)}
                                  style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 8px', background: 'none', border: 'none', borderRadius: '6px', color: ch.completed ? '#888' : '#ddd', cursor: 'pointer', fontSize: '12px', textAlign: 'left', backgroundColor: currExpandedChapter === ch.id ? `rgba(${accentRgb},0.08)` : 'transparent' }}>
                                  <span>{ch.completed ? '+ ' : '> '}{ch.name}</span>
                                  <span style={{ fontSize: '10px', color: '#666' }}>{ch.topics.length} topics</span>
                                </button>
                                {currExpandedChapter === ch.id && (
                                  <div style={{ padding: '4px 0 4px 18px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                    {ch.topics.map(t => (
                                      <button key={t} onClick={() => { setTutorTopic(t); setIsMathMode(currTab === 'math'); setShowCurriculum(false); }}
                                        style={{ padding: '4px 10px', borderRadius: '14px', border: `1px solid rgba(${accentRgb},0.3)`, background: `rgba(${accentRgb},0.06)`, color: currTab === 'cs' ? '#c5b0ff' : '#a0e8d0', cursor: 'pointer', fontSize: '11px', transition: 'all 0.15s' }}
                                        onMouseEnter={e => { (e.target as HTMLElement).style.backgroundColor = `rgba(${accentRgb},0.2)`; }}
                                        onMouseLeave={e => { (e.target as HTMLElement).style.backgroundColor = `rgba(${accentRgb},0.06)`; }}>
                                        {t}
                                      </button>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          )}

          {/* Agent learnings display */}
          {showLearnings && (
            <div style={{ marginBottom: '16px', padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(85,51,255,0.06)', border: '1px solid rgba(85,51,255,0.2)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#b388ff' }}>📚 Agent Learnings</div>
                <button onClick={() => setShowLearnings(false)} style={{ background: 'none', border: '1px solid #5533ff44', borderRadius: '6px', color: '#888', cursor: 'pointer', padding: '4px 10px', fontSize: '11px' }}>
                  Hide
                </button>
              </div>
              {agentLearnings === null ? (
                <div style={{ color: '#888', fontSize: '13px' }}>Loading...</div>
              ) : agentLearnings.length === 0 ? (
                <div style={{ color: '#888', fontSize: '13px' }}>No learnings yet. Use the code agent to build up learnings from debugging sessions.</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '300px', overflowY: 'auto' }}>
                  {agentLearnings.map((l, i) => (
                    <div key={i} style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(0,0,0,0.3)', border: '1px solid #333' }}>
                      <div style={{ fontSize: '12px', color: '#888', marginBottom: '4px' }}>{l.topic} · {new Date(l.timestamp).toLocaleDateString()}</div>
                      <div style={{ fontSize: '13px', color: '#ddd', lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>{l.explanation}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

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

              {/* Solve / Proof input (math mode) */}
              {(tutorProblem.style === 'solve' || tutorProblem.style === 'proof') && (
                <div style={{ marginBottom: '12px' }}>
                  <div style={{ fontSize: '12px', color: '#888', marginBottom: '6px' }}>
                    {tutorProblem.style === 'proof' ? 'Write your proof below:' : 'Enter your answer:'}
                  </div>
                  <textarea value={tutorAnswer} onChange={e => setTutorAnswer(e.target.value)}
                    placeholder={tutorProblem.style === 'proof'
                      ? 'Write your mathematical proof step by step...'
                      : 'Enter your final answer (e.g. 42, 3x+1, -2/3)...'}
                    rows={tutorProblem.style === 'proof' ? 6 : 2}
                    style={{ width: '100%', padding: '12px', borderRadius: '10px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', resize: 'vertical', boxSizing: 'border-box', fontFamily: 'monospace' }} />
                  <button onClick={() => { triggerPulse('submit-math'); handleTutorCheckAnswer(tutorAnswer); }}
                    disabled={tutorLoading || !tutorAnswer.trim() || (tutorFeedback?.solved ?? false)}
                    style={{
                      marginTop: '8px', padding: '8px 20px', borderRadius: '8px', border: 'none', backgroundColor: '#00cc88', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '13px',
                      animation: pulsingBtn === 'submit-math' ? 'greenPulse 0.6s ease-out' : undefined,
                    }}>
                    Submit {tutorProblem.style === 'proof' ? 'Proof' : 'Answer'}
                  </button>
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

              {/* Hint / Next / Steps buttons */}
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <button onClick={() => { triggerPulse('hint'); handleTutorHint(); }} disabled={tutorLoading || (tutorFeedback?.solved ?? false)}
                  style={{
                    padding: '8px 16px', borderRadius: '8px', border: '1px solid #5533ff', backgroundColor: 'transparent', color: '#5533ff', cursor: 'pointer', fontSize: '12px',
                    animation: pulsingBtn === 'hint' ? 'greenPulse 0.6s ease-out' : undefined,
                  }}>
                  Hint
                </button>
                {isMathMode && (
                  <button onClick={() => { triggerPulse('steps'); handleMathSteps(); }}
                    style={{
                      padding: '8px 16px', borderRadius: '8px', border: '1px solid #00cc8866', backgroundColor: 'transparent', color: '#00cc88', cursor: 'pointer', fontSize: '12px',
                      animation: pulsingBtn === 'steps' ? 'greenPulse 0.6s ease-out' : undefined,
                    }}>
                    Step-by-Step
                  </button>
                )}
                {isMathMode && (
                  <button onClick={() => { triggerPulse('graph'); setShowMathGraph(g => { if (!g) handleMathGraph(); return !g; }); }}
                    style={{
                      padding: '8px 16px', borderRadius: '8px', border: '1px solid #ff990066', backgroundColor: showMathGraph ? 'rgba(255,153,0,0.15)' : 'transparent', color: '#ff9900', cursor: 'pointer', fontSize: '12px',
                      animation: pulsingBtn === 'graph' ? 'greenPulse 0.6s ease-out' : undefined,
                    }}>
                    Graph
                  </button>
                )}
                {isMathMode && (
                  <button onClick={() => { triggerPulse('mathviz'); setShowMathViz(v => !v); }}
                    style={{
                      padding: '8px 16px', borderRadius: '8px', border: '1px solid #b388ff66', backgroundColor: showMathViz ? 'rgba(179,136,255,0.15)' : 'transparent', color: '#b388ff', cursor: 'pointer', fontSize: '12px',
                      animation: pulsingBtn === 'mathviz' ? 'greenPulse 0.6s ease-out' : undefined,
                    }}>
                    Math Tools
                  </button>
                )}
                {tutorFeedback?.solved && (
                  <button onClick={() => { triggerPulse('next'); handleTutorGenerate(); }}
                    style={{
                      padding: '8px 16px', borderRadius: '8px', border: 'none', backgroundColor: '#00cc88', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '12px',
                      animation: pulsingBtn === 'next' ? 'greenPulse 0.6s ease-out' : undefined,
                    }}>
                    Next Problem →
                  </button>
                )}
              </div>

              {/* Step-by-step solution display */}
              {mathSteps && mathSteps.length > 0 && (
                <div style={{ marginTop: '12px', padding: '14px', borderRadius: '10px', backgroundColor: 'rgba(0,204,136,0.06)', border: '1px solid rgba(0,204,136,0.2)' }}>
                  <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#00cc88', marginBottom: '10px' }}>Step-by-Step Solution</div>
                  {mathSteps.map((step, i) => (
                    <div key={i} style={{ padding: '6px 0 6px 14px', borderLeft: '2px solid #00cc8844', marginBottom: '6px', fontSize: '13px', color: '#ddd', lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>
                      <span style={{ color: '#00cc88', fontWeight: 'bold', marginRight: '6px' }}>Step {i + 1}:</span>{step}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── Math Interactive Graph ─────────────────────────────────── */}
          {isMathMode && showMathGraph && (
            <div style={{ marginTop: '16px', padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(255,153,0,0.04)', border: '1px solid rgba(255,153,0,0.2)' }}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#ff9900', marginBottom: '10px' }}>📈 Interactive Graph</div>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '8px', alignItems: 'center' }}>
                <span style={{ color: '#00cc88', fontSize: '12px', fontWeight: 'bold' }}>f(x) =</span>
                <input value={mathExpression} onChange={e => setMathExpression(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleMathGraph(); }}
                  placeholder="x^2, sin(x), x^3 - 3*x"
                  style={{ flex: 1, padding: '6px 10px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', fontFamily: 'monospace' }} />
              </div>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', alignItems: 'center' }}>
                <span style={{ color: '#22aaff', fontSize: '12px', fontWeight: 'bold' }}>g(x) =</span>
                <input value={mathExpression2} onChange={e => setMathExpression2(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleMathGraph(); }}
                  placeholder="(optional) second function to compare"
                  style={{ flex: 1, padding: '6px 10px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', fontFamily: 'monospace' }} />
                <button onClick={() => handleMathGraph()}
                  style={{ padding: '6px 14px', borderRadius: '6px', border: 'none', backgroundColor: '#ff9900', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '12px' }}>
                  Plot
                </button>
              </div>
              {mathDerivExpr && (
                <div style={{ fontSize: '12px', color: '#888', marginBottom: '8px' }}>
                  f'(x) = <span style={{ color: '#ff6666', fontFamily: 'monospace' }}>{mathDerivExpr}</span>
                </div>
              )}
              {/* SVG Graph */}
              {mathGraphPoints.length > 0 && (() => {
                const W = 620, H = 380, PAD = 50;
                const validPts = mathGraphPoints.filter(p => isFinite(p.y) && Math.abs(p.y) < 1000);
                const validDeriv = mathDerivPoints.filter(p => isFinite(p.y) && Math.abs(p.y) < 1000);
                const validG = mathGraph2Points.filter(p => isFinite(p.y) && Math.abs(p.y) < 1000);
                if (validPts.length < 2) return <div style={{ color: '#666', fontSize: '12px' }}>No valid points to plot.</div>;
                const xMin = Math.min(...validPts.map(p => p.x));
                const xMax = Math.max(...validPts.map(p => p.x));
                const allY = [...validPts.map(p => p.y), ...validDeriv.map(p => p.y), ...validG.map(p => p.y)];
                const yMin = Math.min(...allY);
                const yMax = Math.max(...allY);
                const xRange = xMax - xMin || 1;
                const yRange = yMax - yMin || 1;
                const sx = (x: number) => PAD + ((x - xMin) / xRange) * (W - 2 * PAD);
                const sy = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - 2 * PAD);
                const toPath = (pts: GraphPoint[]) => pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
                const zeroX = sx(0), zeroY = sy(0);
                // Compute nice tick marks for axes
                const xTickStep = Math.ceil(xRange / 10) || 1;
                const yTickStep = parseFloat((yRange / 8).toPrecision(1)) || 1;
                const xTicks: number[] = [];
                for (let v = Math.ceil(xMin / xTickStep) * xTickStep; v <= xMax; v += xTickStep) xTicks.push(v);
                const yTicks: number[] = [];
                for (let v = Math.ceil(yMin / yTickStep) * yTickStep; v <= yMax; v += yTickStep) yTicks.push(v);
                // Hover
                const hoverPt = mathHoverX !== null ? validPts.reduce((a, b) => Math.abs(b.x - mathHoverX!) < Math.abs(a.x - mathHoverX!) ? b : a, validPts[0]) : null;
                const hoverDeriv = mathHoverX !== null && validDeriv.length > 0 ? validDeriv.reduce((a, b) => Math.abs(b.x - mathHoverX!) < Math.abs(a.x - mathHoverX!) ? b : a, validDeriv[0]) : null;
                // Check if expression looks like it should show integral symbol
                const showIntegral = /integral|∫|antiderivative/i.test(mathExpression) || /integral|∫/i.test(tutorTopic);
                return (
                  <div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222', cursor: 'crosshair' }}
                      onMouseMove={e => {
                        const rect = (e.target as SVGElement).closest('svg')!.getBoundingClientRect();
                        const px = e.clientX - rect.left;
                        const xVal = xMin + ((px - PAD) / (W - 2 * PAD)) * xRange;
                        setMathHoverX(xVal);
                      }}
                      onMouseLeave={() => setMathHoverX(null)}
                    >
                      {/* Grid lines */}
                      {Array.from({ length: 11 }, (_, i) => {
                        const x = PAD + (i / 10) * (W - 2 * PAD);
                        const y = PAD + (i / 10) * (H - 2 * PAD);
                        return <g key={i}>
                          <line x1={x} y1={PAD} x2={x} y2={H - PAD} stroke="#1a1a1a" strokeWidth={1} />
                          <line x1={PAD} y1={y} x2={W - PAD} y2={y} stroke="#1a1a1a" strokeWidth={1} />
                        </g>;
                      })}
                      {/* Axes */}
                      {zeroX >= PAD && zeroX <= W - PAD && <line x1={zeroX} y1={PAD} x2={zeroX} y2={H - PAD} stroke="#555" strokeWidth={1.5} />}
                      {zeroY >= PAD && zeroY <= H - PAD && <line x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY} stroke="#555" strokeWidth={1.5} />}
                      {/* X axis tick labels */}
                      {xTicks.map(v => {
                        const px = sx(v);
                        if (px < PAD + 5 || px > W - PAD - 5) return null;
                        const baseY = zeroY >= PAD && zeroY <= H - PAD ? zeroY : H - PAD;
                        return <g key={`xt${v}`}>
                          <line x1={px} y1={baseY - 3} x2={px} y2={baseY + 3} stroke="#666" strokeWidth={1} />
                          <text x={px} y={baseY + 14} fill="#777" fontSize={9} fontFamily="monospace" textAnchor="middle">{v}</text>
                        </g>;
                      })}
                      {/* Y axis tick labels */}
                      {yTicks.map(v => {
                        const py = sy(v);
                        if (py < PAD + 5 || py > H - PAD - 5) return null;
                        const baseX = zeroX >= PAD && zeroX <= W - PAD ? zeroX : PAD;
                        return <g key={`yt${v}`}>
                          <line x1={baseX - 3} y1={py} x2={baseX + 3} y2={py} stroke="#666" strokeWidth={1} />
                          <text x={baseX - 6} y={py + 3} fill="#777" fontSize={9} fontFamily="monospace" textAnchor="end">{Number.isInteger(v) ? v : v.toFixed(1)}</text>
                        </g>;
                      })}
                      {/* f(x) curve */}
                      <path d={toPath(validPts)} fill="none" stroke="#00cc88" strokeWidth={2} />
                      {/* g(x) second curve */}
                      {validG.length > 1 && <path d={toPath(validG)} fill="none" stroke="#22aaff" strokeWidth={2} />}
                      {/* f'(x) derivative curve */}
                      {validDeriv.length > 1 && <path d={toPath(validDeriv)} fill="none" stroke="#ff6666" strokeWidth={1.5} strokeDasharray="5,3" />}
                      {/* Integral symbol if relevant */}
                      {showIntegral && <text x={PAD + 4} y={PAD + 18} fill="#b388ff" fontSize={22} fontFamily="serif" fontStyle="italic">∫</text>}
                      {/* Hover crosshair + values */}
                      {hoverPt && <>
                        <line x1={sx(hoverPt.x)} y1={PAD} x2={sx(hoverPt.x)} y2={H - PAD} stroke="#ffffff22" strokeWidth={1} />
                        <circle cx={sx(hoverPt.x)} cy={sy(hoverPt.y)} r={4} fill="#00cc88" />
                        {hoverDeriv && <circle cx={sx(hoverDeriv.x)} cy={sy(hoverDeriv.y)} r={3} fill="#ff6666" />}
                        <text x={sx(hoverPt.x) + 8} y={sy(hoverPt.y) - 8} fill="#00cc88" fontSize={11} fontFamily="monospace">
                          ({hoverPt.x.toFixed(2)}, {hoverPt.y.toFixed(2)})
                        </text>
                        {hoverDeriv && <text x={sx(hoverDeriv.x) + 8} y={sy(hoverDeriv.y) + 16} fill="#ff6666" fontSize={10} fontFamily="monospace">
                          f'={hoverDeriv.y.toFixed(2)}
                        </text>}
                      </>}
                      {/* Axis letters */}
                      <text x={W - PAD + 5} y={zeroY >= PAD && zeroY <= H - PAD ? zeroY + 4 : H - PAD + 14} fill="#888" fontSize={11} fontWeight="bold">x</text>
                      <text x={zeroX >= PAD && zeroX <= W - PAD ? zeroX - 14 : PAD - 14} y={PAD - 5} fill="#888" fontSize={11} fontWeight="bold">y</text>
                    </svg>
                    {/* Legend */}
                    <div style={{ display: 'flex', gap: '16px', marginTop: '6px', fontSize: '11px', flexWrap: 'wrap' }}>
                      <span style={{ color: '#00cc88' }}>— f(x) = {mathExpression}</span>
                      {mathExpression2 && validG.length > 0 && <span style={{ color: '#22aaff' }}>— g(x) = {mathExpression2}</span>}
                      {mathDerivExpr && <span style={{ color: '#ff6666' }}>-- f'(x) = {mathDerivExpr}</span>}
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {/* ── Math Visualization Tools ───────────────────────────────── */}
          {isMathMode && showMathViz && (
            <div style={{ marginTop: '16px', padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(179,136,255,0.04)', border: '1px solid rgba(179,136,255,0.2)' }}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#b388ff', marginBottom: '10px' }}>🔧 Math Visualization Tools</div>
              <div style={{ display: 'flex', gap: '6px', marginBottom: '12px', flexWrap: 'wrap' }}>
                {(['unitcircle', 'vector', 'triangle', 'circle', 'matrix', 'normal', 'bezier'] as MathVizType[]).map(t => (
                  <button key={t} onClick={() => setMathVizType(mathVizType === t ? null : t)}
                    style={{ padding: '6px 14px', borderRadius: '6px', border: mathVizType === t ? '1px solid #b388ff' : '1px solid #444', backgroundColor: mathVizType === t ? 'rgba(179,136,255,0.2)' : 'transparent', color: mathVizType === t ? '#b388ff' : '#aaa', cursor: 'pointer', fontSize: '12px' }}>
                    {t === 'unitcircle' ? '🔵 Unit Circle' : t === 'vector' ? '➡️ Vectors' : t === 'triangle' ? '📐 Triangle' : t === 'circle' ? '⭕ Circle' : t === 'matrix' ? '🔢 Matrix Transform' : t === 'normal' ? '📊 Normal Dist' : '〰️ Bezier Curve'}
                  </button>
                ))}
              </div>

              {/* Unit Circle visualization */}
              {mathVizType === 'unitcircle' && (() => {
                const W = 360, H = 360, CX = W / 2, CY = H / 2, R = 140;
                const rad = vizAngle * Math.PI / 180;
                const px = CX + R * Math.cos(rad), py = CY - R * Math.sin(rad);
                const cosVal = Math.cos(rad), sinVal = Math.sin(rad), tanVal = Math.tan(rad);
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>θ =</span>
                      <input type="range" min={0} max={360} value={vizAngle} onChange={e => setVizAngle(+e.target.value)} style={{ flex: 1 }} />
                      <span style={{ color: '#fff', fontFamily: 'monospace', fontSize: '13px', minWidth: '40px' }}>{vizAngle}°</span>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* Grid lines */}
                      <line x1={0} y1={CY} x2={W} y2={CY} stroke="#333" strokeWidth={1} />
                      <line x1={CX} y1={0} x2={CX} y2={H} stroke="#333" strokeWidth={1} />
                      {/* Circle */}
                      <circle cx={CX} cy={CY} r={R} fill="none" stroke="#555" strokeWidth={1.5} />
                      {/* Angle arc */}
                      {vizAngle > 0 && vizAngle < 360 && (
                        <path d={`M${CX + 25},${CY} A25,25 0 ${vizAngle > 180 ? 1 : 0},0 ${CX + 25 * Math.cos(rad)},${CY - 25 * Math.sin(rad)}`} fill="none" stroke="#b388ff" strokeWidth={1.5} />
                      )}
                      {/* Radius line */}
                      <line x1={CX} y1={CY} x2={px} y2={py} stroke="#00cc88" strokeWidth={2} />
                      {/* cos projection (horizontal) */}
                      <line x1={CX} y1={CY} x2={CX + R * cosVal} y2={CY} stroke="#ff9900" strokeWidth={2} strokeDasharray="4,3" />
                      {/* sin projection (vertical) */}
                      <line x1={CX + R * cosVal} y1={CY} x2={px} y2={py} stroke="#22aaff" strokeWidth={2} strokeDasharray="4,3" />
                      {/* Point on circle */}
                      <circle cx={px} cy={py} r={5} fill="#00cc88" />
                      {/* Labels at key positions */}
                      <text x={CX + R + 5} y={CY + 4} fill="#888" fontSize={11}>1</text>
                      <text x={CX - R - 14} y={CY + 4} fill="#888" fontSize={11}>-1</text>
                      <text x={CX + 2} y={CY - R - 4} fill="#888" fontSize={11}>1</text>
                      <text x={CX + 2} y={CY + R + 14} fill="#888" fontSize={11}>-1</text>
                      {/* Coordinate label — clamped to stay inside SVG */}
                      {(() => {
                        // Position label adaptively based on angle quadrant
                        const labelW = 130, labelH = 14;
                        let lx = px + 10, ly = py - 10;
                        // Flip label to the left when point is on the right edge
                        if (px > W - labelW - 10) lx = px - labelW - 4;
                        // Flip label below when point is near top edge
                        if (py < labelH + 10) ly = py + 18;
                        // Clamp to SVG bounds
                        lx = Math.max(4, Math.min(lx, W - labelW - 4));
                        ly = Math.max(14, Math.min(ly, H - 4));
                        return (
                          <text x={lx} y={ly} fill="#00cc88" fontSize={11} fontFamily="monospace">
                            ({cosVal.toFixed(3)}, {sinVal.toFixed(3)})
                          </text>
                        );
                      })()}
                    </svg>
                    <div style={{ display: 'flex', gap: '16px', marginTop: '8px', fontSize: '12px', fontFamily: 'monospace', flexWrap: 'wrap' }}>
                      <span style={{ color: '#ff9900' }}>cos(θ) = {cosVal.toFixed(4)}</span>
                      <span style={{ color: '#22aaff' }}>sin(θ) = {sinVal.toFixed(4)}</span>
                      <span style={{ color: '#ff6666' }}>tan(θ) = {isFinite(tanVal) ? tanVal.toFixed(4) : '∞'}</span>
                      <span style={{ color: '#b388ff' }}>θ = {(rad).toFixed(4)} rad</span>
                    </div>
                  </div>
                );
              })()}

              {/* Vector visualization */}
              {mathVizType === 'vector' && (() => {
                const W = 400, H = 360, CX = W / 2, CY = H / 2, SCALE = 25;
                const vA = vizVectors[0], vB = vizVectors[1];
                const sumX = vA.x + vB.x, sumY = vA.y + vB.y;
                const mag = (v: {x: number; y: number}) => Math.sqrt(v.x * v.x + v.y * v.y);
                const dot = vA.x * vB.x + vA.y * vB.y;
                const angleBetween = Math.acos(Math.min(1, Math.max(-1, dot / (mag(vA) * mag(vB) || 1)))) * 180 / Math.PI;
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '12px', marginBottom: '8px', fontSize: '12px', flexWrap: 'wrap' }}>
                      <span style={{ color: '#00cc88' }}>a⃗ = ({vA.x}, {vA.y})</span>
                      <span style={{ color: '#22aaff' }}>b⃗ = ({vB.x}, {vB.y})</span>
                      <span style={{ color: '#ff9900' }}>a⃗+b⃗ = ({sumX}, {sumY})</span>
                    </div>
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px', flexWrap: 'wrap' }}>
                      {[{ label: 'ax', idx: 0, key: 'x' as const }, { label: 'ay', idx: 0, key: 'y' as const },
                        { label: 'bx', idx: 1, key: 'x' as const }, { label: 'by', idx: 1, key: 'y' as const }].map(({ label, idx, key }) => (
                        <label key={label} style={{ display: 'flex', alignItems: 'center', gap: '4px', color: idx === 0 ? '#00cc88' : '#22aaff', fontSize: '11px' }}>
                          {label}:
                          <input type="number" value={vizVectors[idx][key]}
                            onChange={e => { const next = [...vizVectors]; next[idx] = { ...next[idx], [key]: +e.target.value }; setVizVectors(next); }}
                            style={{ width: '50px', padding: '3px 6px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '12px', fontFamily: 'monospace' }} />
                        </label>
                      ))}
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* Grid */}
                      <line x1={0} y1={CY} x2={W} y2={CY} stroke="#333" strokeWidth={1} />
                      <line x1={CX} y1={0} x2={CX} y2={H} stroke="#333" strokeWidth={1} />
                      {/* Vector a */}
                      <line x1={CX} y1={CY} x2={CX + vA.x * SCALE} y2={CY - vA.y * SCALE} stroke="#00cc88" strokeWidth={2} markerEnd="url(#arrowG)" />
                      {/* Vector b */}
                      <line x1={CX} y1={CY} x2={CX + vB.x * SCALE} y2={CY - vB.y * SCALE} stroke="#22aaff" strokeWidth={2} markerEnd="url(#arrowB)" />
                      {/* Sum vector */}
                      <line x1={CX} y1={CY} x2={CX + sumX * SCALE} y2={CY - sumY * SCALE} stroke="#ff9900" strokeWidth={2} strokeDasharray="5,3" markerEnd="url(#arrowO)" />
                      {/* Parallelogram */}
                      <line x1={CX + vA.x * SCALE} y1={CY - vA.y * SCALE} x2={CX + sumX * SCALE} y2={CY - sumY * SCALE} stroke="#444" strokeWidth={1} strokeDasharray="3,3" />
                      <line x1={CX + vB.x * SCALE} y1={CY - vB.y * SCALE} x2={CX + sumX * SCALE} y2={CY - sumY * SCALE} stroke="#444" strokeWidth={1} strokeDasharray="3,3" />
                      {/* Arrow markers */}
                      <defs>
                        <marker id="arrowG" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#00cc88" /></marker>
                        <marker id="arrowB" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#22aaff" /></marker>
                        <marker id="arrowO" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#ff9900" /></marker>
                      </defs>
                      {/* Labels */}
                      <text x={CX + vA.x * SCALE / 2 - 10} y={CY - vA.y * SCALE / 2 - 6} fill="#00cc88" fontSize={12} fontWeight="bold">a⃗</text>
                      <text x={CX + vB.x * SCALE / 2 + 6} y={CY - vB.y * SCALE / 2 - 6} fill="#22aaff" fontSize={12} fontWeight="bold">b⃗</text>
                    </svg>
                    <div style={{ display: 'flex', gap: '14px', marginTop: '6px', fontSize: '11px', fontFamily: 'monospace', flexWrap: 'wrap' }}>
                      <span style={{ color: '#00cc88' }}>|a⃗| = {mag(vA).toFixed(3)}</span>
                      <span style={{ color: '#22aaff' }}>|b⃗| = {mag(vB).toFixed(3)}</span>
                      <span style={{ color: '#ff9900' }}>|a⃗+b⃗| = {mag({ x: sumX, y: sumY }).toFixed(3)}</span>
                      <span style={{ color: '#ff6666' }}>a⃗·b⃗ = {dot}</span>
                      <span style={{ color: '#b388ff' }}>θ = {angleBetween.toFixed(1)}°</span>
                    </div>
                  </div>
                );
              })()}

              {/* Triangle visualization */}
              {mathVizType === 'triangle' && (() => {
                const W = 400, H = 360, PAD = 40, pts = vizTriangle;
                const maxCoord = Math.max(...pts.map(p => Math.max(Math.abs(p.x), Math.abs(p.y))), 1);
                const scale = (Math.min(W, H) - 2 * PAD) / (2 * maxCoord);
                const tx = (x: number) => W / 2 + x * scale;
                const ty = (y: number) => H / 2 - y * scale;
                const dist = (a: {x: number; y: number}, b: {x: number; y: number}) => Math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2);
                const sideA = dist(pts[1], pts[2]), sideB = dist(pts[0], pts[2]), sideC = dist(pts[0], pts[1]);
                const s = (sideA + sideB + sideC) / 2;
                const area = Math.sqrt(Math.max(0, s * (s - sideA) * (s - sideB) * (s - sideC)));
                // Angles via law of cosines
                const angleAt = (opp: number, a: number, b: number) => Math.acos(Math.min(1, Math.max(-1, (a * a + b * b - opp * opp) / (2 * a * b || 1)))) * 180 / Math.PI;
                const angA = angleAt(sideA, sideB, sideC), angB = angleAt(sideB, sideA, sideC), angC = angleAt(sideC, sideA, sideB);
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px', flexWrap: 'wrap' }}>
                      {['A', 'B', 'C'].map((label, i) => (
                        <span key={label} style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '11px' }}>
                          <span style={{ color: '#b388ff' }}>{label}(</span>
                          <input type="number" value={pts[i].x} onChange={e => { const n = [...pts]; n[i] = { ...n[i], x: +e.target.value }; setVizTriangle(n); }}
                            style={{ width: '40px', padding: '2px 4px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '11px', fontFamily: 'monospace' }} />
                          <span style={{ color: '#888' }}>,</span>
                          <input type="number" value={pts[i].y} onChange={e => { const n = [...pts]; n[i] = { ...n[i], y: +e.target.value }; setVizTriangle(n); }}
                            style={{ width: '40px', padding: '2px 4px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '11px', fontFamily: 'monospace' }} />
                          <span style={{ color: '#b388ff' }}>)</span>
                        </span>
                      ))}
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="#222" strokeWidth={1} />
                      <line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke="#222" strokeWidth={1} />
                      <polygon points={pts.map(p => `${tx(p.x)},${ty(p.y)}`).join(' ')} fill="rgba(179,136,255,0.08)" stroke="#b388ff" strokeWidth={2} />
                      {pts.map((p, i) => (
                        <g key={i}>
                          <circle cx={tx(p.x)} cy={ty(p.y)} r={4} fill="#b388ff" />
                          <text x={tx(p.x) + 8} y={ty(p.y) - 8} fill="#b388ff" fontSize={12} fontWeight="bold">{['A', 'B', 'C'][i]}</text>
                        </g>
                      ))}
                      {/* Side labels */}
                      <text x={(tx(pts[1].x) + tx(pts[2].x)) / 2 + 6} y={(ty(pts[1].y) + ty(pts[2].y)) / 2 - 6} fill="#ff9900" fontSize={10} fontFamily="monospace">a={sideA.toFixed(2)}</text>
                      <text x={(tx(pts[0].x) + tx(pts[2].x)) / 2 - 30} y={(ty(pts[0].y) + ty(pts[2].y)) / 2 + 14} fill="#22aaff" fontSize={10} fontFamily="monospace">b={sideB.toFixed(2)}</text>
                      <text x={(tx(pts[0].x) + tx(pts[1].x)) / 2 + 6} y={(ty(pts[0].y) + ty(pts[1].y)) / 2 + 14} fill="#00cc88" fontSize={10} fontFamily="monospace">c={sideC.toFixed(2)}</text>
                    </svg>
                    <div style={{ display: 'flex', gap: '12px', marginTop: '6px', fontSize: '11px', fontFamily: 'monospace', flexWrap: 'wrap' }}>
                      <span style={{ color: '#ff9900' }}>∠A = {angA.toFixed(1)}°</span>
                      <span style={{ color: '#22aaff' }}>∠B = {angB.toFixed(1)}°</span>
                      <span style={{ color: '#00cc88' }}>∠C = {angC.toFixed(1)}°</span>
                      <span style={{ color: '#b388ff' }}>Area = {area.toFixed(3)}</span>
                      <span style={{ color: '#888' }}>Perimeter = {(sideA + sideB + sideC).toFixed(3)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Circle visualization */}
              {mathVizType === 'circle' && (() => {
                const W = 360, H = 360, SCALE = 20;
                const cx = W / 2 + vizCircle.cx * SCALE, cy = H / 2 - vizCircle.cy * SCALE;
                const rPx = vizCircle.r * SCALE;
                const area = Math.PI * vizCircle.r * vizCircle.r;
                const circumference = 2 * Math.PI * vizCircle.r;
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '8px', flexWrap: 'wrap' }}>
                      {[{ label: 'cx', key: 'cx' as const }, { label: 'cy', key: 'cy' as const }, { label: 'r', key: 'r' as const }].map(({ label, key }) => (
                        <label key={label} style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#b388ff', fontSize: '11px' }}>
                          {label}:
                          <input type="number" value={vizCircle[key]} step={key === 'r' ? 0.5 : 1}
                            onChange={e => setVizCircle({ ...vizCircle, [key]: +e.target.value })}
                            style={{ width: '55px', padding: '3px 6px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '12px', fontFamily: 'monospace' }} />
                        </label>
                      ))}
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="#333" strokeWidth={1} />
                      <line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke="#333" strokeWidth={1} />
                      <circle cx={cx} cy={cy} r={rPx} fill="rgba(179,136,255,0.06)" stroke="#b388ff" strokeWidth={2} />
                      <circle cx={cx} cy={cy} r={3} fill="#ff9900" />
                      {/* Radius line */}
                      <line x1={cx} y1={cy} x2={cx + rPx} y2={cy} stroke="#00cc88" strokeWidth={1.5} strokeDasharray="4,3" />
                      <text x={cx + rPx / 2 - 5} y={cy - 6} fill="#00cc88" fontSize={10} fontFamily="monospace">r={vizCircle.r}</text>
                      <text x={cx + 6} y={cy - 6} fill="#ff9900" fontSize={10} fontFamily="monospace">({vizCircle.cx},{vizCircle.cy})</text>
                    </svg>
                    <div style={{ display: 'flex', gap: '14px', marginTop: '6px', fontSize: '11px', fontFamily: 'monospace', flexWrap: 'wrap' }}>
                      <span style={{ color: '#b388ff' }}>Area = πr² = {area.toFixed(3)}</span>
                      <span style={{ color: '#00cc88' }}>Circumference = 2πr = {circumference.toFixed(3)}</span>
                      <span style={{ color: '#888' }}>Equation: (x−{vizCircle.cx})²+(y−{vizCircle.cy})²={vizCircle.r}²</span>
                    </div>
                  </div>
                );
              })()}

              {/* Matrix transform visualization */}
              {mathVizType === 'matrix' && (() => {
                const W = 400, H = 400, CX = W / 2, CY = H / 2, SCALE = 40;
                const m = vizMatrix;
                // Transform the three standard points
                const transform = (p: {x: number; y: number}) => ({
                  x: m[0][0] * p.x + m[0][1] * p.y,
                  y: m[1][0] * p.x + m[1][1] * p.y,
                });
                const origPts = vizMatrixPoints;
                const transPts = origPts.map(transform);
                const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
                // Eigenvalues for 2x2: λ² - trace·λ + det = 0
                const trace = m[0][0] + m[1][1];
                const disc = trace * trace - 4 * det;
                const eig1 = disc >= 0 ? (trace + Math.sqrt(disc)) / 2 : null;
                const eig2 = disc >= 0 ? (trace - Math.sqrt(disc)) / 2 : null;
                return (
                  <div>
                    {/* Matrix input row */}
                    <div style={{ display: 'flex', gap: '16px', marginBottom: '10px', alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ color: '#b388ff', fontSize: '12px', fontWeight: 600 }}>M =</span>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', padding: '4px 6px', borderLeft: '2px solid #b388ff', borderRight: '2px solid #b388ff', borderRadius: '2px' }}>
                          {[0, 1].map(r => (
                            <div key={r} style={{ display: 'flex', gap: '4px' }}>
                              {[0, 1].map(c => (
                                <input key={`${r}${c}`} type="number" step={0.1} value={m[r][c]}
                                  onChange={e => { const n = m.map(row => [...row]); n[r][c] = +e.target.value; setVizMatrix(n); }}
                                  style={{ width: '60px', padding: '4px 6px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '13px', fontFamily: 'monospace', textAlign: 'center' }} />
                              ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                    {/* Preset buttons */}
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '10px', flexWrap: 'wrap' }}>
                      {[{ label: 'Identity', m: [[1,0],[0,1]] }, { label: 'Rot 45°', m: [[0.707,-0.707],[0.707,0.707]] }, { label: 'Rot 90°', m: [[0,-1],[1,0]] }, { label: 'Shear', m: [[1,1],[0,1]] }, { label: 'Scale 2×', m: [[2,0],[0,2]] }, { label: 'Reflect Y', m: [[-1,0],[0,1]] }, { label: 'Reflect X', m: [[1,0],[0,-1]] }, { label: 'Squeeze', m: [[2,0],[0,0.5]] }].map(p => (
                        <button key={p.label} onClick={() => setVizMatrix(p.m)}
                          style={{ padding: '4px 10px', borderRadius: '4px', border: '1px solid #444', background: 'transparent', color: '#b0b0b0', cursor: 'pointer', fontSize: '11px', transition: 'border-color 0.2s' }}
                          onMouseEnter={e => (e.currentTarget.style.borderColor = '#b388ff')}
                          onMouseLeave={e => (e.currentTarget.style.borderColor = '#444')}>
                          {p.label}
                        </button>
                      ))}
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* Axes */}
                      <line x1={0} y1={CY} x2={W} y2={CY} stroke="#333" strokeWidth={1} />
                      <line x1={CX} y1={0} x2={CX} y2={H} stroke="#333" strokeWidth={1} />
                      {/* Grid */}
                      {[-4,-3,-2,-1,1,2,3,4].map(n => (
                        <g key={n}>
                          <line x1={CX + n * SCALE} y1={0} x2={CX + n * SCALE} y2={H} stroke="#1a1a1a" strokeWidth={1} />
                          <line x1={0} y1={CY + n * SCALE} x2={W} y2={CY + n * SCALE} stroke="#1a1a1a" strokeWidth={1} />
                        </g>
                      ))}
                      {/* Original shape (filled) */}
                      <polygon points={origPts.map(p => `${CX + p.x * SCALE},${CY - p.y * SCALE}`).join(' ')} fill="rgba(0,204,136,0.1)" stroke="#00cc88" strokeWidth={1.5} strokeDasharray="4,3" />
                      {/* Transformed shape */}
                      <polygon points={transPts.map(p => `${CX + p.x * SCALE},${CY - p.y * SCALE}`).join(' ')} fill="rgba(255,136,0,0.12)" stroke="#ff8800" strokeWidth={2} />
                      {/* Transformed basis vectors */}
                      <defs>
                        <marker id="arrowR" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#ff4444" /></marker>
                        <marker id="arrowY" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#ffcc00" /></marker>
                      </defs>
                      <line x1={CX} y1={CY} x2={CX + m[0][0] * SCALE} y2={CY - m[1][0] * SCALE} stroke="#ff4444" strokeWidth={2} markerEnd="url(#arrowR)" />
                      <line x1={CX} y1={CY} x2={CX + m[0][1] * SCALE} y2={CY - m[1][1] * SCALE} stroke="#ffcc00" strokeWidth={2} markerEnd="url(#arrowY)" />
                      {/* Labels */}
                      <text x={CX + m[0][0] * SCALE + 4} y={CY - m[1][0] * SCALE - 6} fill="#ff4444" fontSize={11} fontWeight="bold">ê₁</text>
                      <text x={CX + m[0][1] * SCALE + 4} y={CY - m[1][1] * SCALE - 6} fill="#ffcc00" fontSize={11} fontWeight="bold">ê₂</text>
                      {origPts.map((p, i) => <circle key={`o${i}`} cx={CX + p.x * SCALE} cy={CY - p.y * SCALE} r={3} fill="#00cc88" />)}
                      {transPts.map((p, i) => <circle key={`t${i}`} cx={CX + p.x * SCALE} cy={CY - p.y * SCALE} r={3} fill="#ff8800" />)}
                    </svg>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '6px 14px', marginTop: '8px', fontSize: '11px', fontFamily: 'monospace', padding: '8px', backgroundColor: '#111', borderRadius: '6px', border: '1px solid #222' }}>
                      <span style={{ color: '#b388ff' }}>det(M) = {det.toFixed(3)}</span>
                      <span style={{ color: '#ff9900' }}>tr(M) = {trace.toFixed(3)}</span>
                      {eig1 !== null && <span style={{ color: '#ff4444' }}>λ₁ = {eig1.toFixed(3)}</span>}
                      {eig2 !== null && <span style={{ color: '#ffcc00' }}>λ₂ = {eig2.toFixed(3)}</span>}
                      {disc < 0 && <span style={{ color: '#888' }}>λ = {(trace/2).toFixed(2)} ± {(Math.sqrt(-disc)/2).toFixed(2)}i</span>}
                      <span style={{ color: det > 0 ? '#00cc88' : det < 0 ? '#ff6666' : '#888' }}>
                        {det > 0 ? 'Preserves orientation' : det < 0 ? 'Flips orientation' : 'Singular (det=0)'}
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Normal distribution visualization */}
              {mathVizType === 'normal' && (() => {
                const W = 460, H = 280;
                const mu = vizNormMean, sigma = vizNormStd;
                const mu2 = vizNormMean2, sigma2 = vizNormStd2;
                const sigN = vizNormSigmaShow;
                // Use wider range when comparing two curves
                const allMeans = vizNormCompare ? [mu, mu2] : [mu];
                const allSigmas = vizNormCompare ? [sigma, sigma2] : [sigma];
                const globalMin = Math.min(...allMeans) - 4 * Math.max(...allSigmas, 0.5);
                const globalMax = Math.max(...allMeans) + 4 * Math.max(...allSigmas, 0.5);
                const xMin = globalMin, xMax = globalMax;
                const makePdf = (m: number, s: number) => (x: number) => {
                  const ss = Math.max(s, 0.01);
                  return (1 / (ss * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - m) / ss) ** 2);
                };
                const pdf = makePdf(mu, sigma);
                const pdf2 = makePdf(mu2, sigma2);
                const steps = 140;
                let yMax = 0;
                const allPts: {x: number; y: number}[] = [];
                const allPts2: {x: number; y: number}[] = [];
                for (let i = 0; i <= steps; i++) {
                  const x = xMin + (xMax - xMin) * i / steps;
                  const y = pdf(x);
                  allPts.push({ x, y });
                  if (y > yMax) yMax = y;
                  if (vizNormCompare) {
                    const y2 = pdf2(x);
                    allPts2.push({ x, y: y2 });
                    if (y2 > yMax) yMax = y2;
                  }
                }
                yMax *= 1.1;
                const PAD = 35;
                const toSvgX = (x: number) => PAD + (x - xMin) / (xMax - xMin) * (W - 2 * PAD);
                const toSvgY = (y: number) => (H - PAD) - (y / yMax) * (H - 2 * PAD);
                const pts1Str = allPts.map(p => `${toSvgX(p.x)},${toSvgY(p.y)}`).join(' ');
                const pts2Str = vizNormCompare ? allPts2.map(p => `${toSvgX(p.x)},${toSvgY(p.y)}`).join(' ') : '';
                const fillPath = `M${toSvgX(xMin)},${toSvgY(0)} ` + allPts.map(p => `L${toSvgX(p.x)},${toSvgY(p.y)}`).join(' ') + ` L${toSvgX(xMax)},${toSvgY(0)} Z`;
                const sigmaLines = [-3, -2, -1, 0, 1, 2, 3].map(n => mu + n * sigma);
                // Sigma region area percentages
                const sigmaAreaPct: Record<number, string> = { 1: '68.27', 2: '95.45', 3: '99.73' };
                return (
                  <div>
                    {/* Controls row 1: μ and σ sliders */}
                    <div style={{ display: 'flex', gap: '12px', marginBottom: '6px', fontSize: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#00cc88' }}>
                        μ: <input type="range" min={-5} max={5} step={0.1} value={mu} onChange={e => setVizNormMean(+e.target.value)} />
                        <span style={{ fontFamily: 'monospace', minWidth: '36px' }}>{mu.toFixed(1)}</span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#ff9900' }}>
                        σ: <input type="range" min={0.1} max={4} step={0.1} value={sigma} onChange={e => setVizNormStd(+e.target.value)} />
                        <span style={{ fontFamily: 'monospace', minWidth: '36px' }}>{sigma.toFixed(1)}</span>
                      </label>
                    </div>
                    {/* Controls row 2: σ region selector, area toggle, compare toggle */}
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '6px', fontSize: '11px', flexWrap: 'wrap', alignItems: 'center' }}>
                      <span style={{ color: '#b388ff' }}>Show region:</span>
                      {[1, 2, 3].map(n => (
                        <button key={n} onClick={() => setVizNormSigmaShow(n)}
                          style={{ padding: '2px 8px', borderRadius: '4px', border: `1px solid ${sigN === n ? '#b388ff' : '#444'}`, background: sigN === n ? 'rgba(179,136,255,0.15)' : 'transparent', color: sigN === n ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '11px' }}>
                          ±{n}σ ({sigmaAreaPct[n]}%)
                        </button>
                      ))}
                      <label style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#888', cursor: 'pointer', marginLeft: '4px' }}>
                        <input type="checkbox" checked={vizNormShowArea} onChange={e => setVizNormShowArea(e.target.checked)} style={{ accentColor: '#00cc88' }} />
                        Fill
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#888', cursor: 'pointer' }}>
                        <input type="checkbox" checked={vizNormCompare} onChange={e => setVizNormCompare(e.target.checked)} style={{ accentColor: '#22aaff' }} />
                        Compare
                      </label>
                    </div>
                    {/* Comparison curve controls */}
                    {vizNormCompare && (
                      <div style={{ display: 'flex', gap: '12px', marginBottom: '6px', fontSize: '11px', flexWrap: 'wrap', alignItems: 'center', paddingLeft: '8px', borderLeft: '2px solid #22aaff' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#22aaff' }}>
                          μ₂: <input type="range" min={-5} max={5} step={0.1} value={mu2} onChange={e => setVizNormMean2(+e.target.value)} />
                          <span style={{ fontFamily: 'monospace', minWidth: '36px' }}>{mu2.toFixed(1)}</span>
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#22aaff' }}>
                          σ₂: <input type="range" min={0.1} max={4} step={0.1} value={sigma2} onChange={e => setVizNormStd2(+e.target.value)} />
                          <span style={{ fontFamily: 'monospace', minWidth: '36px' }}>{sigma2.toFixed(1)}</span>
                        </label>
                      </div>
                    )}
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* x-axis */}
                      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#555" strokeWidth={1} />
                      {/* σ lines */}
                      {sigmaLines.map((x, i) => (
                        <g key={i}>
                          <line x1={toSvgX(x)} y1={PAD} x2={toSvgX(x)} y2={H - PAD} stroke={i === 3 ? '#555' : '#222'} strokeWidth={1} strokeDasharray={i === 3 ? undefined : '3,3'} />
                          <text x={toSvgX(x)} y={H - PAD + 14} fill="#888" fontSize={9} textAnchor="middle">
                            {i - 3 === 0 ? 'μ' : `${i - 3 > 0 ? '+' : ''}${i - 3}σ`}
                          </text>
                        </g>
                      ))}
                      {/* Full curve fill */}
                      {vizNormShowArea && <path d={fillPath} fill="rgba(0,204,136,0.08)" />}
                      {/* Nσ shaded region */}
                      {(() => {
                        const lo = mu - sigN * sigma, hi = mu + sigN * sigma;
                        const sigPts = allPts.filter(p => p.x >= lo && p.x <= hi);
                        if (sigPts.length < 2) return null;
                        const d = `M${toSvgX(lo)},${toSvgY(0)} ` + sigPts.map(p => `L${toSvgX(p.x)},${toSvgY(p.y)}`).join(' ') + ` L${toSvgX(hi)},${toSvgY(0)} Z`;
                        return <path d={d} fill="rgba(179,136,255,0.18)" />;
                      })()}
                      {/* Primary curve */}
                      <polyline points={pts1Str} fill="none" stroke="#00cc88" strokeWidth={2} />
                      {/* Comparison curve */}
                      {vizNormCompare && <polyline points={pts2Str} fill="none" stroke="#22aaff" strokeWidth={2} strokeDasharray="6,3" />}
                      {/* Peak markers */}
                      <circle cx={toSvgX(mu)} cy={toSvgY(pdf(mu))} r={3} fill="#00cc88" />
                      {vizNormCompare && <circle cx={toSvgX(mu2)} cy={toSvgY(pdf2(mu2))} r={3} fill="#22aaff" />}
                      {/* σ region boundary lines */}
                      <line x1={toSvgX(mu - sigN * sigma)} y1={PAD} x2={toSvgX(mu - sigN * sigma)} y2={H - PAD} stroke="#b388ff" strokeWidth={1} strokeOpacity={0.5} strokeDasharray="4,2" />
                      <line x1={toSvgX(mu + sigN * sigma)} y1={PAD} x2={toSvgX(mu + sigN * sigma)} y2={H - PAD} stroke="#b388ff" strokeWidth={1} strokeOpacity={0.5} strokeDasharray="4,2" />
                    </svg>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '4px 14px', marginTop: '6px', fontSize: '11px', fontFamily: 'monospace', padding: '6px 8px', backgroundColor: '#111', borderRadius: '6px', border: '1px solid #222' }}>
                      <span style={{ color: '#00cc88' }}>μ = {mu.toFixed(2)}</span>
                      <span style={{ color: '#ff9900' }}>σ = {sigma.toFixed(2)}</span>
                      <span style={{ color: '#b388ff' }}>σ² = {(sigma * sigma).toFixed(3)}</span>
                      <span style={{ color: '#b388ff' }}>P(±{sigN}σ) ≈ {sigmaAreaPct[sigN]}%</span>
                      <span style={{ color: '#888' }}>Peak = {pdf(mu).toFixed(4)}</span>
                      {vizNormCompare && <span style={{ color: '#22aaff' }}>μ₂={mu2.toFixed(2)}, σ₂={sigma2.toFixed(2)}</span>}
                    </div>
                  </div>
                );
              })()}

              {/* Bezier curve visualization */}
              {mathVizType === 'bezier' && (() => {
                const W = 420, H = 300;
                const pts = vizBezierPts;
                const t = vizBezierT;
                // Cubic bezier evaluation
                const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
                const bezierPt = (t: number) => {
                  const p01x = lerp(pts[0].x, pts[1].x, t), p01y = lerp(pts[0].y, pts[1].y, t);
                  const p12x = lerp(pts[1].x, pts[2].x, t), p12y = lerp(pts[1].y, pts[2].y, t);
                  const p23x = lerp(pts[2].x, pts[3].x, t), p23y = lerp(pts[2].y, pts[3].y, t);
                  const p012x = lerp(p01x, p12x, t), p012y = lerp(p01y, p12y, t);
                  const p123x = lerp(p12x, p23x, t), p123y = lerp(p12y, p23y, t);
                  return { x: lerp(p012x, p123x, t), y: lerp(p012y, p123y, t), p01: {x: p01x, y: p01y}, p12: {x: p12x, y: p12y}, p23: {x: p23x, y: p23y}, p012: {x: p012x, y: p012y}, p123: {x: p123x, y: p123y} };
                };
                // Build the curve path
                const curvePts: string[] = [];
                for (let i = 0; i <= 100; i++) {
                  const p = bezierPt(i / 100);
                  curvePts.push(`${p.x},${p.y}`);
                }
                const currentPt = bezierPt(t);
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>t =</span>
                      <input type="range" min={0} max={1} step={0.01} value={t} onChange={e => setVizBezierT(+e.target.value)} style={{ flex: 1 }} />
                      <span style={{ color: '#fff', fontFamily: 'monospace', fontSize: '13px', minWidth: '38px' }}>{t.toFixed(2)}</span>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* Control polygon */}
                      <polyline points={pts.map(p => `${p.x},${p.y}`).join(' ')} fill="none" stroke="#333" strokeWidth={1} strokeDasharray="4,3" />
                      {/* De Casteljau construction lines */}
                      <line x1={currentPt.p01.x} y1={currentPt.p01.y} x2={currentPt.p12.x} y2={currentPt.p12.y} stroke="#444" strokeWidth={1} strokeDasharray="3,3" />
                      <line x1={currentPt.p12.x} y1={currentPt.p12.y} x2={currentPt.p23.x} y2={currentPt.p23.y} stroke="#444" strokeWidth={1} strokeDasharray="3,3" />
                      <line x1={currentPt.p012.x} y1={currentPt.p012.y} x2={currentPt.p123.x} y2={currentPt.p123.y} stroke="#555" strokeWidth={1} strokeDasharray="2,2" />
                      {/* Curve */}
                      <polyline points={curvePts.join(' ')} fill="none" stroke="#00cc88" strokeWidth={2.5} />
                      {/* Control points (draggable look) */}
                      {pts.map((p, i) => (
                        <g key={i}>
                          <circle cx={p.x} cy={p.y} r={6} fill="transparent" stroke="#b388ff" strokeWidth={1.5} />
                          <text x={p.x + 8} y={p.y - 6} fill="#b388ff" fontSize={10}>P{i}</text>
                        </g>
                      ))}
                      {/* Intermediate points */}
                      <circle cx={currentPt.p01.x} cy={currentPt.p01.y} r={3} fill="#ff9900" />
                      <circle cx={currentPt.p12.x} cy={currentPt.p12.y} r={3} fill="#ff9900" />
                      <circle cx={currentPt.p23.x} cy={currentPt.p23.y} r={3} fill="#ff9900" />
                      <circle cx={currentPt.p012.x} cy={currentPt.p012.y} r={3} fill="#22aaff" />
                      <circle cx={currentPt.p123.x} cy={currentPt.p123.y} r={3} fill="#22aaff" />
                      {/* Point on curve */}
                      <circle cx={currentPt.x} cy={currentPt.y} r={5} fill="#ff4444" />
                    </svg>
                    <div style={{ display: 'flex', gap: '6px', marginTop: '6px', flexWrap: 'wrap' }}>
                      {pts.map((p, i) => (
                        <span key={i} style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '10px' }}>
                          <span style={{ color: '#b388ff' }}>P{i}(</span>
                          <input type="number" value={p.x} onChange={e => { const n = [...pts]; n[i] = { ...n[i], x: +e.target.value }; setVizBezierPts(n); }}
                            style={{ width: '40px', padding: '2px 4px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '10px', fontFamily: 'monospace' }} />
                          <span style={{ color: '#888' }}>,</span>
                          <input type="number" value={p.y} onChange={e => { const n = [...pts]; n[i] = { ...n[i], y: +e.target.value }; setVizBezierPts(n); }}
                            style={{ width: '40px', padding: '2px 4px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '10px', fontFamily: 'monospace' }} />
                          <span style={{ color: '#b388ff' }}>)</span>
                        </span>
                      ))}
                    </div>
                    <div style={{ display: 'flex', gap: '14px', marginTop: '4px', fontSize: '11px', fontFamily: 'monospace', flexWrap: 'wrap' }}>
                      <span style={{ color: '#ff4444' }}>B(t) = ({currentPt.x.toFixed(1)}, {currentPt.y.toFixed(1)})</span>
                      <span style={{ color: '#888' }}>Cubic Bézier: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃</span>
                    </div>
                  </div>
                );
              })()}
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
            {/* Practice in Tutor button on agent explanation messages */}
            {isExplanation && activeMode === 'agent' && (
              <div style={{ marginTop: 8 }}>
                <button
                  onClick={() => {
                    // Extract topic from the last agent instruction
                    const rawTopic = lastAgentInstruction.current || 'general';
                    const displayTopic = rawTopic.length > 60 ? rawTopic.slice(0, 60) + '…' : rawTopic;
                    setTutorTopic(displayTopic);
                    // Detect language from selected file extension
                    let detectedLang = tutorLanguage;
                    if (selectedFilePath) {
                      const ext = selectedFilePath.split('.').pop()?.toLowerCase() || '';
                      const langMap: Record<string, string> = { py: 'python', go: 'go', cpp: 'cpp', c: 'c', js: 'javascript', ts: 'typescript', java: 'java', rs: 'rust' };
                      if (langMap[ext]) { detectedLang = langMap[ext]; setTutorLanguage(detectedLang); }
                    }
                    // Switch to tutor mode and auto-fetch learnings with raw topic (no ellipsis) for better matching
                    setActiveMode('tutor');
                    setShowLearnings(true);
                    handleFetchLearnings(rawTopic, detectedLang);
                  }}
                  style={{
                    padding: '6px 12px', borderRadius: '8px',
                    border: '1px solid rgba(0,204,136,0.4)',
                    backgroundColor: 'rgba(0,204,136,0.1)',
                    color: '#00cc88', cursor: 'pointer', fontSize: '12px',
                  }}
                >
                  📚 Practice in Tutor
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
      <div style={{ position: 'fixed', bottom: '28px', right: '40px', zIndex: 50, display: 'flex', alignItems: 'center', gap: '10px' }}>
        {/* LLM connection status indicator + model switcher */}
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '6px' }} data-model-picker>
          <div
            title={llmConnected === null ? 'Checking LLM...' : llmConnected ? `LLM connected: ${llmModel}` : 'LLM offline — queries will fail'}
            style={{
              width: '10px', height: '10px', borderRadius: '50%',
              backgroundColor: llmConnected === null ? '#888' : llmConnected ? '#00cc88' : '#ff3333',
              boxShadow: llmConnected ? '0 0 6px #00cc88' : llmConnected === false ? '0 0 6px #ff3333' : 'none',
              transition: 'all 0.3s', flexShrink: 0,
            }}
          />
          {llmModel && (
            <button
              onClick={() => { if (!modelSwitching) { fetchModels(); setShowModelPicker(p => !p); } }}
              title="Switch model"
              style={{
                background: showModelPicker ? 'rgba(85,51,255,0.25)' : 'rgba(255,255,255,0.07)',
                border: '1px solid rgba(255,255,255,0.2)',
                borderRadius: '8px', padding: '3px 8px',
                color: modelSwitching ? '#888' : '#ccc',
                fontSize: '11px', cursor: modelSwitching ? 'wait' : 'pointer',
                fontFamily: 'monospace', transition: 'all 0.2s',
                display: 'flex', alignItems: 'center', gap: '4px',
              }}
            >
              {modelSwitching ? '⏳' : '⬡'} {llmModel.replace(':latest', '')}
            </button>
          )}
          {showModelPicker && (
            <div
              style={{
                position: 'absolute', bottom: 'calc(100% + 8px)', right: 0,
                background: 'rgba(15,12,30,0.97)', border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '12px', padding: '6px', minWidth: '200px',
                backdropFilter: 'blur(16px)', boxShadow: '0 8px 32px rgba(0,0,0,0.5)', zIndex: 100,
              }}
            >
              <div style={{ color: '#888', fontSize: '10px', padding: '4px 8px 6px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Switch Model</div>
              {availableModels.length === 0
                ? <div style={{ color: '#666', fontSize: '11px', padding: '6px 8px' }}>No models found</div>
                : availableModels.map(m => (
                  <button
                    key={m}
                    onClick={() => handleSwitchModel(m)}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left',
                      background: (m === llmModel || m.replace(':latest','') === llmModel || llmModel === m.replace(':latest',''))
                        ? 'rgba(85,51,255,0.3)' : 'transparent',
                      border: 'none', borderRadius: '8px',
                      color: '#ddd', fontSize: '12px', padding: '7px 10px',
                      cursor: 'pointer', fontFamily: 'monospace',
                      transition: 'background 0.15s',
                    }}
                    onMouseEnter={e => { if (m !== llmModel) (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.08)'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = (m === llmModel) ? 'rgba(85,51,255,0.3)' : 'transparent'; }}
                  >
                    {(m === llmModel || m.replace(':latest', '') === llmModel) && <span style={{ color: '#00cc88', marginRight: '6px' }}>✓</span>}
                    {m.replace(':latest', '')}
                  </button>
                ))
              }
            </div>
          )}
        </div>
        <button
          onClick={() => {
            const modes: ActiveMode[] = ['query', 'agent', 'tutor'];
            const idx = modes.indexOf(activeMode);
            setActiveMode(modes[(idx + 1) % modes.length]);
          }}
          disabled={mode !== 'idle'}
          style={{
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
      </div>

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
          zIndex: 40,
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
            onClick={() => { setUseMultiAgent(p => !p); if (!useMultiAgent) setUseTwoMode(false); }}
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
        {/* Two-Mode agent toggle — auto-routes between Do-It and Explain */}
        {activeMode === 'agent' && (
          <button
            type="button"
            onClick={() => { setUseTwoMode(p => !p); if (!useTwoMode) setUseMultiAgent(false); }}
            disabled={mode !== 'idle'}
            title="Two-mode: auto-classifies difficulty → Do-It (easy) or Explain (hard)"
            style={{
              padding: '10px 10px', borderRadius: '12px', border: '2px solid',
              borderColor: useTwoMode ? '#22aaff' : '#555',
              backgroundColor: useTwoMode ? 'rgba(34,170,255,0.15)' : '#111',
              color: useTwoMode ? '#22aaff' : '#aaa',
              fontSize: '11px', cursor: 'pointer', whiteSpace: 'nowrap',
              transition: 'all 0.3s',
            }}
          >
            🎯{useTwoMode ? ' ✓' : ''}
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

        {/* Process log toggle — next to mic */}
        <button
          type="button"
          onClick={() => setShowProcessPanel(p => !p)}
          title="Toggle backend process logs"
          style={{
            padding: '10px 12px',
            borderRadius: '12px',
            border: '2px solid',
            borderColor: showProcessPanel ? '#ff9900' : '#444',
            backgroundColor: showProcessPanel ? 'rgba(255,153,0,0.15)' : '#111',
            color: showProcessPanel ? '#ff9900' : '#888',
            cursor: 'pointer',
            fontSize: '14px',
            transition: 'all 0.15s',
            position: 'relative',
          }}
        >
          ⚙{processLogs.length > 0 && <span style={{ position: 'absolute', top: '-4px', right: '-4px', backgroundColor: '#ff9900', color: '#000', borderRadius: '50%', width: '16px', height: '16px', fontSize: '9px', fontWeight: 'bold', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{processLogs.length > 99 ? '99+' : processLogs.length}</span>}
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

        {/* Persistent OCR context indicator — shows when image context is active for follow-up questions */}
        {!pendingImageBlob && ocrContext && (
          <div style={{
            padding: '4px 10px', borderRadius: '12px', backgroundColor: 'rgba(0,204,136,0.15)',
            border: '1px solid rgba(0,204,136,0.3)', fontSize: '11px', color: '#00cc88',
            display: 'flex', alignItems: 'center', gap: '6px', marginRight: '4px',
          }}>
            🖼️ Image context active
            <button onClick={() => setOcrContext(null)}
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
        {mode !== 'idle' && (
          <button
            type="button"
            onClick={handleStopAll}
            style={{
              padding: '12px 18px',
              borderRadius: '999px',
              border: '2px solid #ff4444',
              backgroundColor: 'rgba(255,68,68,0.15)',
              color: '#ff4444',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold',
              transition: 'all 0.2s',
              animation: 'pulse 1.5s infinite',
            }}
            title="Stop all running processes"
          >
            ⏹ Stop
          </button>
        )}
        </div>
      </form>
      )}

      {/* ── Process Log Panel (collapsible + expandable) ─────────────── */}
      {showProcessPanel && (
        <div style={{
          position: 'fixed', bottom: '80px', right: '16px', zIndex: 55,
          width: processExpanded ? '700px' : '500px',
          maxWidth: processExpanded ? '70vw' : '45vw',
          height: processExpanded ? '550px' : '300px',
          borderRadius: '12px', overflow: 'hidden',
          backgroundColor: 'rgba(0, 5, 10, 0.92)', backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255,153,0,0.2)', boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
          display: 'flex', flexDirection: 'column',
          transition: 'width 0.25s, height 0.25s, max-width 0.25s',
        }}>
          <div style={{
            padding: '8px 12px', borderBottom: '1px solid #222',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#ff9900' }}>⚙ Backend Process Logs</span>
            <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
              <button onClick={() => setProcessExpanded(p => !p)}
                title={processExpanded ? 'Shrink panel' : 'Expand panel'}
                style={{ background: 'none', border: '1px solid #33333366', borderRadius: '4px', color: '#888', cursor: 'pointer', padding: '2px 8px', fontSize: '10px' }}>
                {processExpanded ? '⊟' : '⊞'}
              </button>
              <button onClick={() => setProcessLogs([])}
                style={{ background: 'none', border: '1px solid #33333366', borderRadius: '4px', color: '#666', cursor: 'pointer', padding: '2px 8px', fontSize: '10px' }}>
                Clear
              </button>
              <button onClick={() => setShowProcessPanel(false)}
                style={{ background: 'none', border: '1px solid #33333366', borderRadius: '4px', color: '#666', cursor: 'pointer', padding: '2px 8px', fontSize: '10px' }}>
                ✕
              </button>
            </div>
          </div>
          <div style={{
            flex: 1, overflowY: 'auto', padding: '6px 10px',
            fontFamily: "'Fira Code', 'JetBrains Mono', Consolas, monospace",
            fontSize: processExpanded ? '12px' : '11px', lineHeight: '1.5',
          }}>
            {processLogs.length === 0 && (
              <div style={{ color: '#555', textAlign: 'center', paddingTop: '40px' }}>
                Waiting for backend activity...
              </div>
            )}
            {processLogs.map((log, i) => {
              const levelColor = log.level === 'ERROR' ? '#ff4444'
                : log.level === 'WARNING' ? '#ffaa33'
                : log.level === 'INFO' ? '#00cc88'
                : '#555';
              return (
                <div key={i} style={{ padding: '2px 0', borderBottom: '1px solid #111', wordBreak: 'break-word' }}>
                  <span style={{ color: '#555', marginRight: '6px' }}>
                    {new Date(log.ts * 1000).toLocaleTimeString()}
                  </span>
                  <span style={{ color: levelColor, fontWeight: 'bold', marginRight: '6px' }}>
                    {log.level}
                  </span>
                  <span style={{ color: '#7d7d7d', marginRight: '6px' }}>
                    [{log.logger}]
                  </span>
                  <span style={{ color: '#ccc' }}>{log.message}</span>
                </div>
              );
            })}
            <div ref={processEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);

// Inject pulse + greenPulse keyframes for animations
const styleEl = document.createElement('style');
styleEl.textContent = `
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(255,68,68,0.5)} 50%{box-shadow:0 0 0 6px rgba(255,68,68,0.0)} }
@keyframes greenPulse { 0%{box-shadow:0 0 0 0 rgba(0,204,136,0.6)} 50%{box-shadow:0 0 0 8px rgba(0,204,136,0)} 100%{box-shadow:0 0 0 0 rgba(0,204,136,0)} }
`;
document.head.appendChild(styleEl);

