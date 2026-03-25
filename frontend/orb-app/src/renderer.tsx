import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import Orb from './Orb';
import './index.css';

type Message    = { role: 'user' | 'ai'; text: string; isDiff?: boolean; imageData?: string };
type OrbMode    = 'idle' | 'querying' | 'agent-processing';
type ActiveMode = 'query' | 'agent' | 'tutor' | 'imagegen';
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
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
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
  const [showDataMenu, setShowDataMenu] = useState(false);
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
  type MathVizType = 'vector' | 'circle' | 'triangle' | 'unitcircle' | 'matrix' | 'normal' | 'bezier' | 'quadratic' | 'calculator' | 'derivative' | 'riemann' | 'taylor' | 'fourier' | 'parametric' | 'regression' | 'montecarlo' | null;
  const [mathVizType, setMathVizType] = useState<MathVizType>(null);
  // Vector: {x, y} endpoints
  const [vizVectors, setVizVectors] = useState<{x: number; y: number; label: string; color: string}[]>([
    { x: 3, y: 2, label: 'a', color: '#00cc88' }, { x: -1, y: 4, label: 'b', color: '#22aaff' }
  ]);
  // 3D vector mode
  const [vizVector3D, setVizVector3D] = useState(false);
  const [vizVec3A, setVizVec3A] = useState<[number, number, number]>([3, 2, 1]);
  const [vizVec3B, setVizVec3B] = useState<[number, number, number]>([-1, 4, 2]);
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

  // ── Curriculum generation state ─────────────────────────────────────────────
  const [currGenStatus, setCurrGenStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle');
  const [currGenProgress, setCurrGenProgress] = useState({ step: 0, total: 0, message: '' });
  const currGenPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  // ── Quadratic viz state ────────────────────────────────────────────────────
  const [vizQuadA, setVizQuadA] = useState(1);
  const [vizQuadB, setVizQuadB] = useState(0);
  const [vizQuadC, setVizQuadC] = useState(-4);
  const [vizQuadResult, setVizQuadResult] = useState<any>(null);

  // ── Scientific calculator state ────────────────────────────────────────────
  const [calcExpr, setCalcExpr] = useState('');
  const [calcResult, setCalcResult] = useState<any>(null);
  const [calcLoading, setCalcLoading] = useState(false);
  const [calcHistory, setCalcHistory] = useState<{expr: string; result: string}[]>([]);

  // ── Derivative animator state ──────────────────────────────────────────────
  const [vizDerivExpr, setVizDerivExpr] = useState('x^2');
  const [vizDerivX, setVizDerivX] = useState(1);

  // ── Riemann sum state ──────────────────────────────────────────────────────
  const [vizRiemannExpr, setVizRiemannExpr] = useState('x^2');
  const [vizRiemannN, setVizRiemannN] = useState(10);
  const [vizRiemannA, setVizRiemannA] = useState(0);
  const [vizRiemannB, setVizRiemannB] = useState(2);
  const [vizRiemannMethod, setVizRiemannMethod] = useState<'left' | 'right' | 'midpoint'>('left');

  // ── Taylor series state ────────────────────────────────────────────────────
  const [vizTaylorFn, setVizTaylorFn] = useState<'sin' | 'cos' | 'exp'>('sin');
  const [vizTaylorN, setVizTaylorN] = useState(3);

  // ── Fourier series state ───────────────────────────────────────────────────
  const [vizFourierWave, setVizFourierWave] = useState<'square' | 'sawtooth' | 'triangle'>('square');
  const [vizFourierN, setVizFourierN] = useState(5);

  // ── Parametric curves state ────────────────────────────────────────────────
  const [vizParamXExpr, setVizParamXExpr] = useState('cos(t)');
  const [vizParamYExpr, setVizParamYExpr] = useState('sin(t)');
  const [vizParamT, setVizParamT] = useState(3.14);

  // ── Regression state ───────────────────────────────────────────────────────
  const [vizRegPoints, setVizRegPoints] = useState<{x: number; y: number}[]>([
    {x:1,y:2},{x:2,y:4},{x:3,y:5},{x:4,y:4},{x:5,y:5}
  ]);
  const [vizRegShowResiduals, setVizRegShowResiduals] = useState(true);

  // ── Monte Carlo Pi state ───────────────────────────────────────────────────
  const [vizMonteN, setVizMonteN] = useState(500);
  const [vizMontePoints, setVizMontePoints] = useState<{x: number; y: number; inside: boolean}[]>([]);

  // ── TTS state ──────────────────────────────────────────────────────────────
  const [ttsLoading, setTtsLoading] = useState(false);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);

  // ── Image generation state ─────────────────────────────────────────────────
  const [imageGenLoading, setImageGenLoading] = useState(false);
  const [imageGenPrompt, setImageGenPrompt] = useState('');
  const [imageGenResult, setImageGenResult] = useState<string | null>(null);
  const [showImageGenDialog, setShowImageGenDialog] = useState(false);
  // ── Image generation page state ────────────────────────────────────────────
  type ImageModel = { name: string; filename: string; path: string; size_mb: number; type: 'checkpoint' | 'lora' };
  const [imageModels, setImageModels] = useState<ImageModel[]>([]);
  const [activeImageModel, setActiveImageModel] = useState<string | null>(null);
  const [selectedImageModel, setSelectedImageModel] = useState<string>('');
  const [selectedLoras, setSelectedLoras] = useState<string[]>(() => {
    try { const s = localStorage.getItem('aion_selectedLoras'); return s ? JSON.parse(s) : []; } catch { return []; }
  });
  const [imageGenMode, setImageGenMode] = useState<'normal' | 'explicit'>('normal');
  const [imageGenSteps, setImageGenSteps] = useState(35);
  const [imageGenCfg, setImageGenCfg] = useState(7.5);
  const [imageGenSeed, setImageGenSeed] = useState(-1);
  const [imageGenWidth, setImageGenWidth] = useState(1024);
  const [imageGenHeight, setImageGenHeight] = useState(1024);
  const [imageGenNegative, setImageGenNegative] = useState('');
  const [imageGenMeta, setImageGenMeta] = useState<any>(null);
  const [imageGenHistory, setImageGenHistory] = useState<any[]>([]);
  const [imageGenFeedback, setImageGenFeedback] = useState('');
  const [imageGenAnimated, setImageGenAnimated] = useState(false);
  const [imageGenFrames, setImageGenFrames] = useState(16);
  // ── New ImageStudio state ──────────────────────────────────────────────────
  const [imageGenArtStyle, setImageGenArtStyle] = useState(() => {
    try { return localStorage.getItem('aion_imageGenArtStyle') || 'anime'; } catch { return 'anime'; }
  });
  const [imageGenFrameStrength, setImageGenFrameStrength] = useState(0.35);
  const [imageGenFps, setImageGenFps] = useState(8);
  const [imageGenOutputFormat, setImageGenOutputFormat] = useState<'gif'|'mp4'|'both'>('gif');
  const [artStyles, setArtStyles] = useState<{name:string; prefix:string}[]>([]);
  const [imageGenSubTab, setImageGenSubTab] = useState<'generate'|'animate'|'storyboard'|'upscale'|'train'|'video'>('generate');
  const [storyboardDescs, setStoryboardDescs] = useState('');
  const [animJobs, setAnimJobs] = useState<any[]>([]);
  // ── Animation anti-deterioration & dynamic controls ────────────────────────
  const [animReferenceBlend, setAnimReferenceBlend] = useState(0.3);
  const [animStrengthCurve, setAnimStrengthCurve] = useState<'constant'|'ease_in'|'pulse'>('constant');
  const [animMotionIntensity, setAnimMotionIntensity] = useState(0.5);
  // ── UI layout ──────────────────────────────────────────────────────────────
  const [modelsExpanded, setModelsExpanded] = useState(false);
  // ── GPU monitor & progress tracking ────────────────────────────────────────
  const [gpuInfo, setGpuInfo] = useState<{ vram_used_mb: number; vram_total_mb: number; vram_percent: number; device_name: string; available?: boolean; error?: string } | null>(null);
  const [genProgress, setGenProgress] = useState<{ active: boolean; type: string; current_step: number; total_steps: number; current_frame: number; total_frames: number; message: string }>({ active: false, type: 'idle', current_step: 0, total_steps: 0, current_frame: 0, total_frames: 0, message: '' });
  const progressPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // ── Video generation (WAN) ─────────────────────────────────────────────────
  const [videoRefImage, setVideoRefImage] = useState('');
  const [videoFrames, setVideoFrames] = useState(81);
  const [videoFps, setVideoFps] = useState(16);
  const [videoSteps, setVideoSteps] = useState(30);
  const [videoCfg, setVideoCfg] = useState(5.0);
  const [videoActionLora, setVideoActionLora] = useState('');
  const [actionLoras, setActionLoras] = useState<any[]>([]);
  const [isVideoResult, setIsVideoResult] = useState(false);
  const [interruptedJobs, setInterruptedJobs] = useState<any[]>([]);
  const [trainingStatus, setTrainingStatus] = useState<any>(null);
  const [trainingImageDir, setTrainingImageDir] = useState('');
  const [trainingName, setTrainingName] = useState('');
  const [trainingType, setTrainingType] = useState<'style'|'character'>('style');
  const [trainingCritique, setTrainingCritique] = useState<any>(null);
  const [upscaleImagePath, setUpscaleImagePath] = useState('');
  const [upscaleScale, setUpscaleScale] = useState(2.0);
  const [upscaleTileSize, setUpscaleTileSize] = useState(768);
  // ── Vocab expansion & tag marking ──────────────────────────────────────────
  const [vocabSuggestions, setVocabSuggestions] = useState<Record<string, string[]>>({});
  const [tagRating, setTagRating] = useState<string>('');  // danbooru/e621 tag category
  // ── Token counter state ────────────────────────────────────────────────────
  const [promptTokenCount, setPromptTokenCount] = useState<number>(0);
  const [promptTokenMethod, setPromptTokenMethod] = useState<string>('estimate');
  const tokenDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // ── LoRA browser & trigger word management ─────────────────────────────────
  const [loraCategories, setLoraCategories] = useState<Record<string, {name:string; filename:string; path:string; size_mb:number; trigger_words:string[]}[]>>({});
  const [loraBrowserOpen, setLoraBrowserOpen] = useState(false);
  const [loraBrowserCategory, setLoraBrowserCategory] = useState('styles');
  const [triggerWordInput, setTriggerWordInput] = useState('');
  const [editingTriggerLora, setEditingTriggerLora] = useState<string | null>(null);
  const [loraSearchQuery, setLoraSearchQuery] = useState('');
  const [loraSearchResults, setLoraSearchResults] = useState<any[]>([]);
  const [loraSearchHistoryResults, setLoraSearchHistoryResults] = useState<any[]>([]);
  // ── Image preview mode ─────────────────────────────────────────────────────
  const [previewResult, setPreviewResult] = useState<string | null>(null);
  const [previewSeed, setPreviewSeed] = useState<number | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  // ── Image lightbox with zoom+pan ───────────────────────────────────────────
  const [lightboxZoom, setLightboxZoom] = useState(1);
  const [lightboxPan, setLightboxPan] = useState({ x: 0, y: 0 });
  const lightboxDragging = useRef(false);
  const lightboxDragStart = useRef({ x: 0, y: 0 });
  // ── Curriculum chapter resume state ────────────────────────────────────────
  const [currChapterData, setCurrChapterData] = useState<any>(null);
  const [currChapterProgress, setCurrChapterProgress] = useState<any>(null);
  const [currChapterIndex, setCurrChapterIndex] = useState(0);
  const [currChapterAnswers, setCurrChapterAnswers] = useState<Record<string, any>>({});
  const [currChapterActive, setCurrChapterActive] = useState(false);
  const [currChapterInput, setCurrChapterInput] = useState('');
  const [currChapterFeedback, setCurrChapterFeedback] = useState<any>(null);
  const [currChapterLoading, setCurrChapterLoading] = useState(false);

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

  // ── GPU monitor + generation progress polling ─────────────────────────────
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const [gpuRes, progRes] = await Promise.all([
          fetch('http://localhost:8000/system/gpu'),
          fetch('http://localhost:8000/generate/progress'),
        ]);
        if (gpuRes.ok) setGpuInfo(await gpuRes.json());
        if (progRes.ok) setGenProgress(await progRes.json());
      } catch { /* backend down */ }
    }, imageGenLoading ? 1000 : 5000);  // poll faster during generation
    progressPollRef.current = poll;
    return () => clearInterval(poll);
  }, [imageGenLoading]);

  // ── Fetch resumable video jobs when video tab shown or generation finishes ──
  useEffect(() => {
    if (imageGenSubTab !== 'video') return;
    const fetchResumable = async () => {
      try {
        const res = await fetch('http://localhost:8000/generate/video/queue');
        if (res.ok) {
          const data = await res.json();
          setInterruptedJobs(data.resumable_jobs || data.interrupted_jobs || []);
        }
      } catch {}
    };
    fetchResumable();
  }, [imageGenSubTab, imageGenLoading]);

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
    if (activeMode === 'imagegen') { fetchImageModels(); fetchImageHistory(); fetchArtStyles(); fetchAnimJobs(); fetchLoraCategories(); }
  }, [activeMode]);

  // Persist selectedLoras and artStyle to localStorage
  useEffect(() => {
    try { localStorage.setItem('aion_selectedLoras', JSON.stringify(selectedLoras)); } catch {}
  }, [selectedLoras]);
  useEffect(() => {
    try { localStorage.setItem('aion_imageGenArtStyle', imageGenArtStyle); } catch {}
  }, [imageGenArtStyle]);

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

  // ── Curriculum chapter resume handlers ──────────────────────────────────
  const handleOpenChapter = async (subjectId: string, chapterId: string) => {
    setCurrChapterLoading(true);
    setCurrChapterFeedback(null);
    setCurrChapterInput('');
    try {
      const [chapterRes, progressRes] = await Promise.all([
        fetch(`http://localhost:8000/curriculum/chapter/${subjectId}/${chapterId}`),
        fetch(`http://localhost:8000/curriculum/progress/${subjectId}/${chapterId}`),
      ]);
      const chapterData = await chapterRes.json();
      const progressData = await progressRes.json();
      if (chapterData.status === 'ok') {
        setCurrChapterData(chapterData);
        const answers = progressData.answers || {};
        const startIdx = progressData.current_index || 0;
        setCurrChapterAnswers(answers);
        setCurrChapterIndex(startIdx);
        setCurrChapterActive(true);
        setShowCurriculum(false);
      }
    } catch (err) {
      console.error('Failed to load chapter:', err);
    } finally {
      setCurrChapterLoading(false);
    }
  };

  const handleChapterAnswer = async () => {
    if (!currChapterData || currChapterLoading) return;
    const problem = currChapterData.problems?.[currChapterIndex];
    if (!problem) return;
    setCurrChapterLoading(true);
    setCurrChapterFeedback(null);

    try {
      // Ensure session is registered in backend (may have been lost on restart)
      await fetch(`http://localhost:8000/curriculum/restore-session/${currChapterData.subject_id}/${currChapterData.chapter_id}/${currChapterIndex}`, { method: 'POST' });

      const endpoint = currChapterData.is_math ? 'http://localhost:8000/math/check' : 'http://localhost:8000/tutor/check';
      const payload: any = { session_id: problem.session_id, answer: currChapterInput };
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      const correct = data.correct ?? false;
      const prevAttempts = currChapterAnswers[String(currChapterIndex)]?.attempts ?? 0;
      const newAnswers = {
        ...currChapterAnswers,
        [String(currChapterIndex)]: {
          correct: correct || (currChapterAnswers[String(currChapterIndex)]?.correct ?? false),
          user_answer: currChapterInput,
          attempts: prevAttempts + 1,
        },
      };
      setCurrChapterAnswers(newAnswers);
      setCurrChapterFeedback(data);

      // Auto-save progress
      await fetch('http://localhost:8000/curriculum/progress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_id: currChapterData.subject_id,
          chapter_id: currChapterData.chapter_id,
          current_index: currChapterIndex,
          answers: newAnswers,
          completed: correct && currChapterIndex >= (currChapterData.total_problems ?? 0) - 1,
        }),
      });

      // Refresh gamification profile (XP is awarded inside check_answer)
      if (correct) {
        try {
          const gp = await fetch('http://localhost:8000/gamification/profile');
          const gd = await gp.json();
          if (gd.status === 'ok') setGamifProfile(gd.profile);
        } catch {}
      }
    } catch (err) {
      setCurrChapterFeedback({ correct: false, feedback: 'Error checking answer.' });
    } finally {
      setCurrChapterLoading(false);
    }
  };

  const handleChapterNext = async () => {
    const nextIdx = currChapterIndex + 1;
    if (nextIdx >= (currChapterData?.total_problems ?? 0)) return;
    setCurrChapterIndex(nextIdx);
    setCurrChapterInput('');
    setCurrChapterFeedback(null);
    // Save progress
    await fetch('http://localhost:8000/curriculum/progress', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        subject_id: currChapterData.subject_id,
        chapter_id: currChapterData.chapter_id,
        current_index: nextIdx,
        answers: currChapterAnswers,
        completed: false,
      }),
    });
  };

  const handleCloseChapter = () => {
    setCurrChapterActive(false);
    setCurrChapterData(null);
    setCurrChapterProgress(null);
    setCurrChapterFeedback(null);
    setCurrChapterInput('');
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

// ── Fast noisy preview ──────────────────────────────────────────────────
const handleImagePreview = async () => {
  if (!imageGenPrompt.trim() || previewLoading || imageGenLoading) return;
  setPreviewLoading(true);
  setPreviewResult(null);
  setPreviewSeed(null);
  setImageGenResult(null);
  setMode('querying');

  try {
    const body: any = {
      prompt: (tagRating ? tagRating + ', ' : '') + imageGenPrompt.trim(),
      width: imageGenWidth,
      height: imageGenHeight,
      mode: imageGenMode,
      steps: 12,
      guidance_scale: imageGenCfg,
      seed: imageGenSeed,
      art_style: imageGenArtStyle,
    };
    if (selectedImageModel) body.model = selectedImageModel;
    if (selectedLoras.length > 0) body.loras = selectedLoras;
    if (imageGenNegative.trim()) body.negative_prompt = imageGenNegative.trim();

    const res = await fetch('http://localhost:8000/generate/preview/quick', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      let errMsg = 'Preview failed';
      try { const errData = await res.json(); errMsg = errData.error || errData.detail || errMsg; }
      catch { try { errMsg = await res.text() || errMsg; } catch {} }
      setImageGenResult(`Error: ${errMsg}`);
      return;
    }

    const contentType = res.headers.get('content-type') ?? '';
    if (contentType.includes('image')) {
      const metaHeader = res.headers.get('X-Generation-Meta');
      if (metaHeader) {
        try {
          const meta = JSON.parse(metaHeader);
          setPreviewSeed(meta.seed ?? null);
          setImageGenMeta(meta);
        } catch {}
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setPreviewResult(url);
    } else {
      const data = await res.json();
      if (data.status === 'error') {
        setImageGenResult(`Error: ${data.error || 'Preview failed'}`);
      }
    }
  } catch (err: any) {
    setImageGenResult(`Error: ${err.message}`);
  } finally {
    setPreviewLoading(false);
    setMode('idle');
  }
};

// ── Full render from preview (uses the preview seed) ────────────────────
const handleFullRenderFromPreview = async () => {
  if (previewSeed === null) return;
  setImageGenSeed(previewSeed);
  setPreviewResult(null);
  // Trigger normal generate with the locked seed
  setImageGenAnimated(false);
  setImageGenLoading(true);
  setImageGenResult(null);
  setImageGenMeta(null);
  setMode('querying');

  try {
    const body: any = {
      prompt: (tagRating ? tagRating + ', ' : '') + imageGenPrompt.trim(),
      width: imageGenWidth,
      height: imageGenHeight,
      mode: imageGenMode,
      steps: imageGenSteps,
      guidance_scale: imageGenCfg,
      seed: previewSeed,
      art_style: imageGenArtStyle,
    };
    if (selectedImageModel) body.model = selectedImageModel;
    if (selectedLoras.length > 0) body.loras = selectedLoras;
    if (imageGenNegative.trim()) body.negative_prompt = imageGenNegative.trim();

    const res = await fetch('http://localhost:8000/generate/image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      let errMsg = 'Generation failed';
      try { const errData = await res.json(); errMsg = errData.error || errData.detail || errMsg; }
      catch { try { errMsg = await res.text() || errMsg; } catch {} }
      setImageGenResult(`Error: ${errMsg}`);
      return;
    }

    const contentType = res.headers.get('content-type') ?? '';
    if (contentType.includes('image')) {
      const metaHeader = res.headers.get('X-Generation-Meta');
      if (metaHeader) {
        try { setImageGenMeta(JSON.parse(metaHeader)); } catch {}
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setImageGenResult(url);
    } else {
      const data = await res.json();
      if (data.status === 'ok') {
        setImageGenResult(data.path);
        setImageGenMeta(data);
      } else {
        setImageGenResult(`Error: ${data.error || 'Generation failed'}`);
      }
    }
  } catch (err: any) {
    setImageGenResult(`Error: ${err.message}`);
  } finally {
    setImageGenLoading(false);
    setMode('idle');
    fetchImageHistory();
  }
};

const handleImageGenerate = async (animatedOverride?: boolean) => {
  if (!imageGenPrompt.trim() || imageGenLoading) return;
  const isAnimated = animatedOverride !== undefined ? animatedOverride : imageGenAnimated;
  setImageGenAnimated(isAnimated);
  setImageGenLoading(true);
  setImageGenResult(null);
  setIsVideoResult(false);
  setImageGenMeta(null);
  setPreviewResult(null);
  setMode('querying');  // activate orb animation during generation

  const endpoint = isAnimated ? '/generate/animated' : '/generate/image';

  try {
    const body: any = {
      prompt: (tagRating ? tagRating + ', ' : '') + imageGenPrompt.trim(),
      width: imageGenWidth,
      height: imageGenHeight,
      mode: imageGenMode,
      steps: imageGenSteps,
      guidance_scale: imageGenCfg,
      seed: imageGenSeed,
    };
    if (selectedImageModel) body.model = selectedImageModel;
    if (selectedLoras.length > 0) body.loras = selectedLoras;
    if (imageGenNegative.trim()) body.negative_prompt = imageGenNegative.trim();
    body.art_style = imageGenArtStyle;
    if (isAnimated) {
      body.num_frames = imageGenFrames;
      body.frame_strength = imageGenFrameStrength;
      body.fps = imageGenFps;
      body.output_format = imageGenOutputFormat;
      body.reference_blend = animReferenceBlend;
      body.strength_curve = animStrengthCurve;
      body.motion_intensity = animMotionIntensity;
      if (storyboardDescs.trim()) {
        body.storyboard = storyboardDescs.split('\n').filter((s: string) => s.trim());
      }
    }

    const res = await fetch(`http://localhost:8000${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      let errMsg = 'Generation failed';
      try { const errData = await res.json(); errMsg = errData.error || errData.detail || errMsg; }
      catch { try { errMsg = await res.text() || errMsg; } catch {} }
      setImageGenResult(`Error: ${errMsg}`);
      return;
    }

    const contentType = res.headers.get('content-type') ?? '';
    if (contentType.includes('image') || contentType.includes('gif') || contentType.includes('video')) {
      // Parse metadata from custom header
      const metaHeader = res.headers.get('X-Generation-Meta') || res.headers.get('X-Animation-Meta');
      if (metaHeader) {
        try { setImageGenMeta(JSON.parse(metaHeader)); } catch {}
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setImageGenResult(url);
    } else {
      const data = await res.json();
      if (data.status === 'ok') {
        setImageGenResult(data.description || data.path);
        setImageGenMeta(data);
      } else {
        setImageGenResult(`Error: ${data.error || 'Generation failed'}`);
      }
    }
  } catch (err: any) {
    setImageGenResult(`Error: ${err.message}`);
  } finally {
    setImageGenLoading(false);
    setMode('idle');  // deactivate orb
    fetchImageHistory();  // refresh history after every generation attempt
  }
};

// Fetch image generation models
const fetchImageModels = async () => {
  try {
    const res = await fetch('http://localhost:8000/generate/models');
    const data = await res.json();
    if (data.status === 'ok') {
      setImageModels(data.models || []);
      setActiveImageModel(data.active);
      if (!selectedImageModel && data.models?.length > 0) {
        const checkpoints = data.models.filter((m: ImageModel) => m.type === 'checkpoint');
        if (checkpoints.length > 0) setSelectedImageModel(checkpoints[0].name);
      }
    }
  } catch {}
};

// Fetch image generation history
const fetchImageHistory = async () => {
  try {
    const res = await fetch('http://localhost:8000/generate/history?limit=20');
    const data = await res.json();
    if (data.status === 'ok') setImageGenHistory(data.history || []);
  } catch {}
};

// Submit feedback for a generation
const submitImageFeedback = async (index: number, feedback: string) => {
  try {
    await fetch('http://localhost:8000/generate/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ generation_index: index, feedback }),
    });
    setImageGenFeedback('');
    fetchImageHistory();
  } catch {}
};

// Delete an image model
const deleteImageModel = async (modelPath: string) => {
  if (!confirm('Delete this model permanently?')) return;
  try {
    await fetch('http://localhost:8000/generate/models/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath }),
    });
    fetchImageModels();
  } catch {}
};

// Fetch available art styles
const fetchArtStyles = async () => {
  try {
    const res = await fetch('http://localhost:8000/generate/styles');
    const data = await res.json();
    if (data.status === 'ok') setArtStyles(data.styles || []);
  } catch {}
};

// Fetch LoRA categories with trigger words
const fetchLoraCategories = async () => {
  try {
    const res = await fetch('http://localhost:8000/generate/loras/categories');
    const data = await res.json();
    if (data.status === 'ok') setLoraCategories(data.categories || {});
  } catch {}
};

// Save trigger words for a LoRA
const saveTriggerWords = async (loraName: string, words: string) => {
  try {
    const wordList = words.split(',').map((w: string) => w.trim()).filter(Boolean);
    await fetch('http://localhost:8000/generate/loras/trigger-words', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lora_name: loraName, trigger_words: wordList }),
    });
    setEditingTriggerLora(null);
    setTriggerWordInput('');
    fetchLoraCategories();
  } catch {}
};

// Search LoRAs across all categories
const searchLoras = async (query: string) => {
  if (!query.trim()) { setLoraSearchResults([]); setLoraSearchHistoryResults([]); return; }
  try {
    const res = await fetch(`http://localhost:8000/generate/characters/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();
    if (data.status === 'ok') {
      setLoraSearchResults(data.loras || []);
      setLoraSearchHistoryResults(data.history_matches || []);
    }
  } catch {}
};

// ── Token counter ────────────────────────────────────────────────────────
const updateTokenCount = (text: string) => {
  if (tokenDebounceRef.current) clearTimeout(tokenDebounceRef.current);
  // Immediate client-side estimate
  const words = text.trim().split(/\s+/).filter(Boolean);
  const estimate = Math.max(0, Math.round(words.length * 1.3));
  setPromptTokenCount(estimate);
  setPromptTokenMethod('estimate');
  // Debounced backend call for exact count
  if (text.trim()) {
    tokenDebounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch('http://localhost:8000/generate/tokenize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: text.trim() }),
        });
        if (res.ok) {
          const data = await res.json();
          setPromptTokenCount(data.token_count ?? estimate);
          setPromptTokenMethod(data.method ?? 'estimate');
        }
      } catch {}
    }, 600);
  } else {
    setPromptTokenCount(0);
  }
};

// Insert trigger word into prompt
const insertTriggerWord = (word: string) => {
  setImageGenPrompt(prev => {
    const trimmed = prev.trim();
    return trimmed ? trimmed + ', ' + word : word;
  });
};

// Fetch animation jobs
const fetchAnimJobs = async () => {
  try {
    const res = await fetch('http://localhost:8000/generate/animated/jobs');
    const data = await res.json();
    if (data.status === 'ok') setAnimJobs(data.jobs || []);
  } catch {}
};

// Fetch training status
const fetchTrainingStatus = async () => {
  try {
    const res = await fetch('http://localhost:8000/generate/train/status');
    const data = await res.json();
    setTrainingStatus(data);
  } catch {}
};

// Critique training dataset
const handleCritiqueDataset = async () => {
  if (!trainingImageDir.trim()) return;
  try {
    const res = await fetch('http://localhost:8000/generate/train/critique', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_dir: trainingImageDir, training_type: trainingType }),
    });
    const data = await res.json();
    setTrainingCritique(data);
  } catch {}
};

// Start LoRA training
const handleStartTraining = async () => {
  if (!trainingName.trim() || !trainingImageDir.trim()) return;
  try {
    const res = await fetch('http://localhost:8000/generate/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: trainingName, image_dir: trainingImageDir,
        training_type: trainingType,
        base_model: selectedImageModel || undefined,
      }),
    });
    const data = await res.json();
    setTrainingStatus(data);
  } catch {}
};

// Upscale image
const handleUpscale = async () => {
  if (!upscaleImagePath.trim()) return;
  setImageGenLoading(true);
  try {
    const res = await fetch('http://localhost:8000/generate/upscale', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: upscaleImagePath, scale: upscaleScale,
        tile_size: upscaleTileSize,
        model: selectedImageModel || undefined,
      }),
    });
    const contentType = res.headers.get('content-type') ?? '';
    if (contentType.includes('image')) {
      const blob = await res.blob();
      setImageGenResult(URL.createObjectURL(blob));
    } else {
      const data = await res.json();
      if (data.status === 'error') setImageGenResult(`Error: ${data.error}`);
    }
  } catch (err: any) {
    setImageGenResult(`Error: ${err.message}`);
  } finally {
    setImageGenLoading(false);
  }
};

// Generate storyboard preview
const handleStoryboardPreview = async () => {
  if (!imageGenPrompt.trim()) return;
  setImageGenLoading(true);
  try {
    const body: any = {
      prompt: imageGenPrompt.trim(),
      num_frames: imageGenFrames,
      width: 384, height: 384,
      seed: imageGenSeed,
    };
    if (selectedImageModel) body.model = selectedImageModel;
    if (storyboardDescs.trim()) {
      body.storyboard_descriptions = storyboardDescs.split('\n').filter((s: string) => s.trim());
    }
    const res = await fetch('http://localhost:8000/generate/storyboard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const contentType = res.headers.get('content-type') ?? '';
    if (contentType.includes('image')) {
      const blob = await res.blob();
      setImageGenResult(URL.createObjectURL(blob));
    } else {
      const data = await res.json();
      if (data.status === 'error') setImageGenResult(`Error: ${data.error}`);
    }
  } catch (err: any) {
    setImageGenResult(`Error: ${err.message}`);
  } finally {
    setImageGenLoading(false);
  }
};

// Fetch vocabulary expansion suggestions
const fetchVocabExpansion = async (text: string) => {
  if (!text.trim()) { setVocabSuggestions({}); return; }
  try {
    const res = await fetch('http://localhost:8000/generate/vocab/expand', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.status === 'ok') setVocabSuggestions(data.suggestions || {});
  } catch {}
};

// Cancel running generation
const handleCancelGeneration = async () => {
  try {
    await fetch('http://localhost:8000/generate/cancel', { method: 'POST' });
    // Also cancel any in-flight fetch
    if (globalAbortRef.current) {
      globalAbortRef.current.abort();
      globalAbortRef.current = null;
    }
  } catch {}
  setImageGenLoading(false);
  setMode('idle');
};

// Free all VRAM (unload SD pipelines + evict Ollama models)
const [vramFlushing, setVramFlushing] = React.useState(false);
const handleFlushVram = async () => {
  setVramFlushing(true);
  try {
    await fetch('http://localhost:8000/generate/vram/flush', { method: 'POST' });
  } catch {}
  setVramFlushing(false);
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
        {/* Data Management Dropdown */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setShowDataMenu(prev => !prev)}
            style={{ padding: '6px 12px', borderRadius: '6px', backgroundColor: '#333', color: '#fff', border: '1px solid #555', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}
            disabled={mode !== 'idle'}
          >
            📂 Data ▾
          </button>
          {showDataMenu && (
            <>
            <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 99 }} onClick={() => setShowDataMenu(false)} />
            <div style={{
              position: 'absolute', top: '100%', left: 0, marginTop: '4px',
              backgroundColor: '#1a1a1a', border: '1px solid #444', borderRadius: '8px',
              minWidth: '180px', padding: '4px', zIndex: 100,
              boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
            }}>
              <button
                onClick={async () => {
                  setShowDataMenu(false);
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
                style={{ display: 'block', width: '100%', padding: '8px 12px', borderRadius: '6px', backgroundColor: 'transparent', color: '#fff', border: 'none', textAlign: 'left', cursor: 'pointer', fontSize: '13px' }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#333')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                📁 Open Data Folder
              </button>
              <button
                onClick={async () => {
                  setShowDataMenu(false);
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
                style={{ display: 'block', width: '100%', padding: '8px 12px', borderRadius: '6px', backgroundColor: 'transparent', color: '#fff', border: 'none', textAlign: 'left', cursor: 'pointer', fontSize: '13px' }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#333')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                📥 Batch Ingest All
              </button>
              <button
                onClick={async () => {
                  setShowDataMenu(false);
                  setMode('querying');
                  setMessages(prev => [...prev, { role: 'user', text: 'Re-ingesting all documents (force rebuild)...' }]);
                  try {
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
                    let done = false;
                    let logCursor = 0;
                    while (!done) {
                      await new Promise(r => setTimeout(r, 3000));
                      const statusRes = await fetch(`http://localhost:8000/reingest/status?log_cursor=${logCursor}`);
                      const statusJson = await statusRes.json();
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
                style={{ display: 'block', width: '100%', padding: '8px 12px', borderRadius: '6px', backgroundColor: 'transparent', color: '#fff', border: 'none', textAlign: 'left', cursor: 'pointer', fontSize: '13px' }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#333')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                🔄 Re-ingest
              </button>
              <button
                onClick={() => { setShowDataMenu(false); (document.getElementById('pdf-upload') as HTMLInputElement).click(); }}
                style={{ display: 'block', width: '100%', padding: '8px 12px', borderRadius: '6px', backgroundColor: 'transparent', color: '#fff', border: 'none', textAlign: 'left', cursor: 'pointer', fontSize: '13px' }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#333')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                📄 Upload PDF
              </button>
            </div>
            </>
          )}
        </div>

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
          position: 'fixed', top: '60px', left: '16px', right: '16px', bottom: '16px',
          zIndex: 35, overflowY: 'auto',
          padding: '24px 28px', borderRadius: '20px',
          backgroundColor: 'rgba(0, 20, 15, 0.94)', backdropFilter: 'blur(25px)',
          border: '1px solid rgba(0,204,136,0.3)', boxShadow: '0 12px 40px rgba(0,0,0,0.6)',
        }}>
          {/* Setup bar */}
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '10px', alignItems: 'center' }}>
            {/* Math / CS mode toggle — first for visibility */}
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
              {isMathMode ? '∑ Math' : '🖥 CS'}
            </button>
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
              📚 Browse Topics
            </button>
            <select value={tutorDifficulty} onChange={e => setTutorDifficulty(e.target.value as any)} disabled={tutorLoading}
              style={{ padding: '8px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', opacity: tutorLoading ? 0.5 : 1, cursor: tutorLoading ? 'not-allowed' : 'pointer' }}>
              <option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option>
            </select>
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
          </div>

          {/* ── Math / CS Tools Bar — always visible when a problem exists ── */}
          {tutorProblem && (
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '10px', padding: '8px 12px', borderRadius: '10px', backgroundColor: 'rgba(0,204,136,0.05)', border: '1px solid rgba(0,204,136,0.15)' }}>
              <button onClick={() => { triggerPulse('hint'); handleTutorHint(); }} disabled={tutorLoading || (tutorFeedback?.solved ?? false)}
                style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid #5533ff', backgroundColor: 'transparent', color: '#5533ff', cursor: 'pointer', fontSize: '12px', animation: pulsingBtn === 'hint' ? 'greenPulse 0.6s ease-out' : undefined }}>
                💡 Hint
              </button>
              {isMathMode && (
                <button onClick={() => { triggerPulse('steps'); handleMathSteps(); }}
                  style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid #00cc8866', backgroundColor: 'transparent', color: '#00cc88', cursor: 'pointer', fontSize: '12px', animation: pulsingBtn === 'steps' ? 'greenPulse 0.6s ease-out' : undefined }}>
                  📝 Step-by-Step
                </button>
              )}
              {isMathMode && (
                <button onClick={() => { triggerPulse('graph'); setShowMathGraph(g => { if (!g) handleMathGraph(); return !g; }); }}
                  style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid #ff990066', backgroundColor: showMathGraph ? 'rgba(255,153,0,0.15)' : 'transparent', color: '#ff9900', cursor: 'pointer', fontSize: '12px', animation: pulsingBtn === 'graph' ? 'greenPulse 0.6s ease-out' : undefined }}>
                  📊 Graph
                </button>
              )}
              {isMathMode && (
                <button onClick={() => { triggerPulse('mathviz'); setShowMathViz(v => !v); }}
                  style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid #b388ff66', backgroundColor: showMathViz ? 'rgba(179,136,255,0.15)' : 'transparent', color: '#b388ff', cursor: 'pointer', fontSize: '12px', animation: pulsingBtn === 'mathviz' ? 'greenPulse 0.6s ease-out' : undefined }}>
                  🧮 Math Tools
                </button>
              )}
              {tutorFeedback?.solved && (
                <button onClick={() => { triggerPulse('next'); handleTutorGenerate(); }}
                  style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', backgroundColor: '#00cc88', color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '12px', animation: pulsingBtn === 'next' ? 'greenPulse 0.6s ease-out' : undefined }}>
                  Next Problem →
                </button>
              )}
            </div>
          )}

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
                                  <div style={{ padding: '4px 0 4px 18px' }}>
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '8px' }}>
                                      {ch.topics.map(t => (
                                        <button key={t} onClick={() => { setTutorTopic(t); setIsMathMode(currTab === 'math'); setShowCurriculum(false); }}
                                          style={{ padding: '4px 10px', borderRadius: '14px', border: `1px solid rgba(${accentRgb},0.3)`, background: `rgba(${accentRgb},0.06)`, color: currTab === 'cs' ? '#c5b0ff' : '#a0e8d0', cursor: 'pointer', fontSize: '11px', transition: 'all 0.15s' }}
                                          onMouseEnter={e => { (e.target as HTMLElement).style.backgroundColor = `rgba(${accentRgb},0.2)`; }}
                                          onMouseLeave={e => { (e.target as HTMLElement).style.backgroundColor = `rgba(${accentRgb},0.06)`; }}>
                                          {t}
                                        </button>
                                      ))}
                                    </div>
                                    {/* Generate entire chapter */}
                                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
                                      <button
                                        onClick={async () => {
                                          setCurrGenStatus('running');
                                          setCurrGenProgress({ step: 0, total: 0, message: 'Starting...' });
                                          try {
                                            const res = await fetch('http://localhost:8000/curriculum/generate', {
                                              method: 'POST',
                                              headers: { 'Content-Type': 'application/json' },
                                              body: JSON.stringify({ subject_id: subjId, chapter_id: ch.id, is_math: currTab === 'math', language: tutorLanguage }),
                                            });
                                            const d = await res.json();
                                            if (d.status === 'ok' && d.cached) {
                                              setCurrGenStatus('done');
                                              setCurrGenProgress({ step: 0, total: 0, message: `Loaded ${d.problems?.length ?? 0} cached problems` });
                                              return;
                                            }
                                            if (d.status === 'started' || d.status === 'already_running') {
                                              // Poll for progress
                                              if (currGenPollRef.current) clearInterval(currGenPollRef.current);
                                              currGenPollRef.current = setInterval(async () => {
                                                try {
                                                  const sr = await fetch('http://localhost:8000/curriculum/generate/status');
                                                  const sd = await sr.json();
                                                  setCurrGenProgress({ step: sd.step ?? 0, total: sd.total ?? 0, message: sd.message ?? '' });
                                                  if (sd.status === 'done' || sd.status === 'error') {
                                                    setCurrGenStatus(sd.status);
                                                    if (currGenPollRef.current) clearInterval(currGenPollRef.current);
                                                  }
                                                } catch {}
                                              }, 1500);
                                            }
                                          } catch (err) {
                                            setCurrGenStatus('error');
                                            setCurrGenProgress({ step: 0, total: 0, message: String(err) });
                                          }
                                        }}
                                        disabled={currGenStatus === 'running'}
                                        style={{
                                          padding: '5px 12px', borderRadius: '6px', border: 'none',
                                          backgroundColor: currGenStatus === 'running' ? '#555' : accentColor,
                                          color: currGenStatus === 'running' ? '#888' : '#000',
                                          fontWeight: 'bold', cursor: currGenStatus === 'running' ? 'wait' : 'pointer',
                                          fontSize: '11px',
                                        }}
                                      >
                                        {currGenStatus === 'running' ? '⏳ Generating...' : '📝 Generate Chapter'}
                                      </button>
                                      <button
                                        onClick={async () => {
                                          setCurrGenStatus('running');
                                          setCurrGenProgress({ step: 0, total: 0, message: 'Force regenerating...' });
                                          try {
                                            const res = await fetch('http://localhost:8000/curriculum/generate', {
                                              method: 'POST',
                                              headers: { 'Content-Type': 'application/json' },
                                              body: JSON.stringify({ subject_id: subjId, chapter_id: ch.id, is_math: currTab === 'math', language: tutorLanguage, force: true }),
                                            });
                                            const d = await res.json();
                                            if (d.status === 'started' || d.status === 'already_running') {
                                              if (currGenPollRef.current) clearInterval(currGenPollRef.current);
                                              currGenPollRef.current = setInterval(async () => {
                                                try {
                                                  const sr = await fetch('http://localhost:8000/curriculum/generate/status');
                                                  const sd = await sr.json();
                                                  setCurrGenProgress({ step: sd.step ?? 0, total: sd.total ?? 0, message: sd.message ?? '' });
                                                  if (sd.status === 'done' || sd.status === 'error') {
                                                    setCurrGenStatus(sd.status);
                                                    if (currGenPollRef.current) clearInterval(currGenPollRef.current);
                                                  }
                                                } catch {}
                                              }, 1500);
                                            }
                                          } catch (err) {
                                            setCurrGenStatus('error');
                                            setCurrGenProgress({ step: 0, total: 0, message: String(err) });
                                          }
                                        }}
                                        disabled={currGenStatus === 'running'}
                                        title="Delete existing chapter and regenerate all problems with correct answers"
                                        style={{
                                          padding: '5px 12px', borderRadius: '6px', border: '1px solid rgba(255,150,0,0.4)',
                                          backgroundColor: currGenStatus === 'running' ? '#333' : 'rgba(255,150,0,0.1)',
                                          color: currGenStatus === 'running' ? '#888' : '#ff9900',
                                          fontWeight: 'bold', cursor: currGenStatus === 'running' ? 'wait' : 'pointer',
                                          fontSize: '11px',
                                        }}
                                      >
                                        🔄 Regenerate
                                      </button>
                                      <button
                                        onClick={() => handleOpenChapter(subjId, ch.id)}
                                        disabled={currChapterLoading}
                                        style={{
                                          padding: '5px 12px', borderRadius: '6px',
                                          border: `1px solid rgba(${accentRgb},0.4)`,
                                          backgroundColor: `rgba(${accentRgb},0.08)`,
                                          color: currTab === 'cs' ? '#c5b0ff' : '#a0e8d0',
                                          fontWeight: 'bold', cursor: currChapterLoading ? 'wait' : 'pointer',
                                          fontSize: '11px',
                                        }}
                                      >
                                        {currChapterLoading ? '⏳...' : '▶ Resume Chapter'}
                                      </button>
                                      {currGenStatus === 'running' && currGenProgress.total > 0 && (
                                        <div style={{ flex: 1, minWidth: '120px' }}>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#888', marginBottom: '2px' }}>
                                            <span>{currGenProgress.message}</span>
                                            <span>{currGenProgress.step}/{currGenProgress.total}</span>
                                          </div>
                                          <div style={{ height: '4px', borderRadius: '2px', backgroundColor: '#222', overflow: 'hidden' }}>
                                            <div style={{
                                              height: '100%', borderRadius: '2px',
                                              backgroundColor: accentColor,
                                              width: `${Math.round((currGenProgress.step / currGenProgress.total) * 100)}%`,
                                              transition: 'width 0.3s',
                                            }} />
                                          </div>
                                        </div>
                                      )}
                                      {currGenStatus === 'done' && (
                                        <span style={{ color: '#00cc88', fontSize: '11px' }}>✓ {currGenProgress.message}</span>
                                      )}
                                      {currGenStatus === 'error' && (
                                        <span style={{ color: '#ff4444', fontSize: '11px' }}>✗ {currGenProgress.message}</span>
                                      )}
                                    </div>
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

          {/* ── Curriculum Chapter Viewer ─────────────────────────────────── */}
          {currChapterActive && currChapterData && (() => {
            const problems = currChapterData.problems || [];
            const problem = problems[currChapterIndex];
            const totalProblems = currChapterData.total_problems || problems.length;
            const answeredCount = Object.values(currChapterAnswers).filter((a: any) => a.correct).length;
            const currentAnswer = currChapterAnswers[String(currChapterIndex)];
            const accentCol = currChapterData.is_math ? '#00cc88' : '#b388ff';

            return (
              <div style={{ marginBottom: '16px', padding: '16px', borderRadius: '12px', backgroundColor: 'rgba(0,0,0,0.3)', border: `1px solid ${accentCol}44` }}>
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <div>
                    <span style={{ fontSize: '15px', fontWeight: 'bold', color: accentCol }}>
                      📚 {currChapterData.chapter_name}
                    </span>
                    <span style={{ fontSize: '12px', color: '#888', marginLeft: '10px' }}>
                      {currChapterData.subject_name}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span style={{ fontSize: '11px', color: '#888' }}>
                      ✓ {answeredCount}/{totalProblems} · Problem {currChapterIndex + 1}/{totalProblems}
                    </span>
                    <button onClick={handleCloseChapter}
                      style={{ background: 'none', border: '1px solid #55555555', borderRadius: '6px', color: '#888', cursor: 'pointer', padding: '4px 10px', fontSize: '11px' }}>
                      ✕ Close
                    </button>
                  </div>
                </div>

                {/* Progress bar */}
                <div style={{ height: '6px', borderRadius: '3px', backgroundColor: '#222', overflow: 'hidden', marginBottom: '14px' }}>
                  <div style={{
                    height: '100%', borderRadius: '3px', transition: 'width 0.3s',
                    width: `${Math.round((answeredCount / totalProblems) * 100)}%`,
                    background: `linear-gradient(90deg, ${accentCol}, ${accentCol}88)`,
                  }} />
                </div>

                {/* Problem navigation dots */}
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginBottom: '14px' }}>
                  {problems.map((_: any, i: number) => {
                    const ans = currChapterAnswers[String(i)];
                    const isCurrent = i === currChapterIndex;
                    return (
                      <button key={i} onClick={() => { setCurrChapterIndex(i); setCurrChapterInput(''); setCurrChapterFeedback(null); }}
                        style={{
                          width: '24px', height: '24px', borderRadius: '50%', border: 'none',
                          fontSize: '10px', fontWeight: 'bold', cursor: 'pointer',
                          backgroundColor: ans?.correct ? accentCol
                            : ans && !ans.correct ? '#ff444444'
                            : isCurrent ? '#333' : '#1a1a1a',
                          color: ans?.correct ? '#000' : isCurrent ? '#fff' : '#555',
                          outline: isCurrent ? `2px solid ${accentCol}` : 'none',
                          outlineOffset: '2px',
                        }}
                      >
                        {i + 1}
                      </button>
                    );
                  })}
                </div>

                {problem ? (
                  <>
                    {/* Lesson card (collapsible) */}
                    {problem.lesson && (
                      <details style={{ marginBottom: '12px' }}>
                        <summary style={{ cursor: 'pointer', color: accentCol, fontSize: '13px', fontWeight: 'bold', marginBottom: '8px' }}>
                          📖 {problem.lesson.title || 'Lesson'}
                        </summary>
                        <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(0,0,0,0.3)', fontSize: '13px', color: '#ccc', lineHeight: '1.6' }}>
                          <p style={{ margin: '0 0 8px' }}>{problem.lesson.explanation}</p>
                          {problem.lesson.rules?.map((r: string, ri: number) => (
                            <div key={ri} style={{ marginBottom: '4px', paddingLeft: '12px', borderLeft: `2px solid ${accentCol}44` }}>
                              <span style={{ color: '#aaa', fontSize: '12px' }}>{r}</span>
                            </div>
                          ))}
                          {problem.lesson.example_code && (
                            <div style={{ marginTop: '8px', padding: '8px', borderRadius: '6px', backgroundColor: '#0a0a0a', fontSize: '12px', color: '#eee', whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                              {problem.lesson.example_code}
                            </div>
                          )}
                        </div>
                      </details>
                    )}

                    {/* Question */}
                    <div style={{ marginBottom: '12px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0,0,0,0.25)' }}>
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '6px' }}>
                        <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '10px', backgroundColor: problem.difficulty === 'easy' ? '#00cc8822' : problem.difficulty === 'medium' ? '#ff990022' : '#ff444422', color: problem.difficulty === 'easy' ? '#00cc88' : problem.difficulty === 'medium' ? '#ff9900' : '#ff4444', textTransform: 'uppercase', fontWeight: 'bold' }}>
                          {problem.difficulty}
                        </span>
                        <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '10px', backgroundColor: '#33333366', color: '#888' }}>{problem.style}</span>
                        <span style={{ fontSize: '11px', color: '#666' }}>{problem.topic}</span>
                      </div>
                      <div style={{ fontSize: '14px', color: '#eee', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                        {problem.question}
                      </div>
                    </div>

                    {/* Answer area */}
                    {!currentAnswer?.correct && (
                      <div style={{ marginBottom: '12px' }}>
                        {problem.style === 'mcq' && problem.options ? (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {problem.options.map((opt: string, oi: number) => (
                              <button key={oi}
                                onClick={() => setCurrChapterInput(opt)}
                                style={{
                                  padding: '8px 14px', borderRadius: '8px', textAlign: 'left',
                                  border: currChapterInput === opt ? `2px solid ${accentCol}` : '1px solid #333',
                                  backgroundColor: currChapterInput === opt ? `${accentCol}15` : '#111',
                                  color: currChapterInput === opt ? '#fff' : '#ccc',
                                  cursor: 'pointer', fontSize: '13px', transition: 'all 0.15s',
                                }}
                              >
                                {opt}
                              </button>
                            ))}
                          </div>
                        ) : (
                          <textarea
                            value={currChapterInput}
                            onChange={e => setCurrChapterInput(e.target.value)}
                            placeholder="Type your answer..."
                            rows={3}
                            style={{
                              width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid #333',
                              backgroundColor: '#0a0a0a', color: '#eee', fontSize: '13px', fontFamily: problem.style === 'code' ? 'monospace' : 'inherit',
                              resize: 'vertical',
                            }}
                          />
                        )}
                        <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                          <button
                            onClick={handleChapterAnswer}
                            disabled={!currChapterInput.trim() || currChapterLoading}
                            style={{
                              padding: '8px 20px', borderRadius: '8px', border: 'none',
                              backgroundColor: accentCol, color: '#000', fontWeight: 'bold',
                              cursor: (!currChapterInput.trim() || currChapterLoading) ? 'not-allowed' : 'pointer',
                              fontSize: '13px', opacity: (!currChapterInput.trim() || currChapterLoading) ? 0.5 : 1,
                            }}
                          >
                            {currChapterLoading ? '⏳ Checking...' : 'Submit Answer'}
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Feedback */}
                    {currChapterFeedback && (
                      <div style={{
                        padding: '12px', borderRadius: '8px', marginBottom: '12px',
                        backgroundColor: currChapterFeedback.correct ? 'rgba(0,204,136,0.08)' : 'rgba(255,68,68,0.08)',
                        border: `1px solid ${currChapterFeedback.correct ? '#00cc8844' : '#ff444444'}`,
                      }}>
                        <div style={{ fontSize: '14px', fontWeight: 'bold', color: currChapterFeedback.correct ? '#00cc88' : '#ff4444', marginBottom: '6px' }}>
                          {currChapterFeedback.correct ? '✓ Correct!' : '✗ Not quite'}
                        </div>
                        {currChapterFeedback.feedback && (
                          <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>
                            {currChapterFeedback.feedback}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Already answered correctly */}
                    {currentAnswer?.correct && (
                      <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(0,204,136,0.08)', border: '1px solid #00cc8844', marginBottom: '12px' }}>
                        <span style={{ color: '#00cc88', fontSize: '13px', fontWeight: 'bold' }}>✓ Answered correctly</span>
                        {currentAnswer.user_answer && <span style={{ color: '#888', fontSize: '12px', marginLeft: '8px' }}>— {currentAnswer.user_answer.slice(0, 100)}</span>}
                      </div>
                    )}

                    {/* Navigation */}
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                      {currChapterIndex > 0 && (
                        <button onClick={() => { setCurrChapterIndex(currChapterIndex - 1); setCurrChapterInput(''); setCurrChapterFeedback(null); }}
                          style={{ padding: '6px 14px', borderRadius: '6px', border: '1px solid #333', backgroundColor: 'transparent', color: '#888', cursor: 'pointer', fontSize: '12px' }}>
                          ← Previous
                        </button>
                      )}
                      {currChapterIndex < totalProblems - 1 && (currentAnswer?.correct || currChapterFeedback?.correct) && (
                        <button onClick={handleChapterNext}
                          style={{ padding: '6px 14px', borderRadius: '6px', border: 'none', backgroundColor: accentCol, color: '#000', fontWeight: 'bold', cursor: 'pointer', fontSize: '12px' }}>
                          Next Problem →
                        </button>
                      )}
                      {answeredCount === totalProblems && (
                        <span style={{ color: '#ffcc00', fontSize: '13px', fontWeight: 'bold', alignSelf: 'center' }}>
                          🏆 Chapter Complete!
                        </span>
                      )}
                    </div>
                  </>
                ) : (
                  <div style={{ color: '#888', fontSize: '13px' }}>Problem data unavailable for index {currChapterIndex}.</div>
                )}
              </div>
            );
          })()}

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

              {/* Hint / Next / Steps buttons moved to toolbar above */}

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
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#b388ff' }}>🔧 Math Visualization Tools</div>
                <button onClick={() => setShowMathViz(false)} style={{ background: 'none', border: '1px solid #b388ff44', borderRadius: '6px', color: '#888', cursor: 'pointer', padding: '4px 10px', fontSize: '11px' }}>Close</button>
              </div>
              {/* Tab bar */}
              <div style={{ display: 'flex', gap: '0px', marginBottom: '16px', borderBottom: '2px solid #222', overflowX: 'auto', flexWrap: 'wrap' }}>
                {([
                  { key: 'unitcircle', label: '🔵 Unit Circle' },
                  { key: 'vector', label: '➡️ Vectors' },
                  { key: 'triangle', label: '📐 Triangle' },
                  { key: 'circle', label: '⭕ Circle' },
                  { key: 'matrix', label: '🔢 Matrix Transform' },
                  { key: 'normal', label: '📊 Normal Dist' },
                  { key: 'bezier', label: '〰️ Bezier' },
                  { key: 'quadratic', label: '📈 Quadratic' },
                  { key: 'derivative', label: '∫ Derivative' },
                  { key: 'riemann', label: '∫ Riemann' },
                  { key: 'taylor', label: '∫ Taylor' },
                  { key: 'fourier', label: '∫ Fourier' },
                  { key: 'parametric', label: '∫ Parametric' },
                  { key: 'regression', label: '🎲 Regression' },
                  { key: 'montecarlo', label: '🎲 Monte Carlo π' },
                  { key: 'calculator', label: '🔬 Calculator' },
                ] as {key: MathVizType; label: string}[]).map(t => (
                  <button key={t.key} onClick={() => setMathVizType(mathVizType === t.key ? null : t.key)}
                    style={{
                      padding: '7px 12px', border: 'none', borderBottom: mathVizType === t.key ? '2px solid #b388ff' : '2px solid transparent',
                      backgroundColor: 'transparent', color: mathVizType === t.key ? '#b388ff' : '#777',
                      cursor: 'pointer', fontSize: '11px', fontWeight: mathVizType === t.key ? 'bold' : 'normal',
                      whiteSpace: 'nowrap', transition: 'all 0.2s', marginBottom: '-2px',
                    }}
                    onMouseEnter={e => { if (mathVizType !== t.key) (e.target as HTMLElement).style.color = '#b388ff'; }}
                    onMouseLeave={e => { if (mathVizType !== t.key) (e.target as HTMLElement).style.color = '#777'; }}>
                    {t.label}
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
                const is3D = vizVector3D;

                if (!is3D) {
                  // ── 2D mode (original) ──
                  const vA = vizVectors[0], vB = vizVectors[1];
                  const sumX = vA.x + vB.x, sumY = vA.y + vB.y;
                  const mag = (v: {x: number; y: number}) => Math.sqrt(v.x * v.x + v.y * v.y);
                  const dot = vA.x * vB.x + vA.y * vB.y;
                  const angleBetween = Math.acos(Math.min(1, Math.max(-1, dot / (mag(vA) * mag(vB) || 1)))) * 180 / Math.PI;
                  return (
                    <div>
                      <div style={{ display: 'flex', gap: '8px', marginBottom: '8px', alignItems: 'center' }}>
                        <button onClick={() => setVizVector3D(false)} style={{ padding: '4px 10px', borderRadius: '4px', border: '1px solid #00cc88', backgroundColor: 'rgba(0,204,136,0.2)', color: '#00cc88', fontSize: '11px', cursor: 'pointer' }}>2D</button>
                        <button onClick={() => setVizVector3D(true)} style={{ padding: '4px 10px', borderRadius: '4px', border: '1px solid #555', backgroundColor: 'transparent', color: '#888', fontSize: '11px', cursor: 'pointer' }}>3D</button>
                      </div>
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
                        <line x1={0} y1={CY} x2={W} y2={CY} stroke="#333" strokeWidth={1} />
                        <line x1={CX} y1={0} x2={CX} y2={H} stroke="#333" strokeWidth={1} />
                        <line x1={CX} y1={CY} x2={CX + vA.x * SCALE} y2={CY - vA.y * SCALE} stroke="#00cc88" strokeWidth={2} markerEnd="url(#arrowG)" />
                        <line x1={CX} y1={CY} x2={CX + vB.x * SCALE} y2={CY - vB.y * SCALE} stroke="#22aaff" strokeWidth={2} markerEnd="url(#arrowB)" />
                        <line x1={CX} y1={CY} x2={CX + sumX * SCALE} y2={CY - sumY * SCALE} stroke="#ff9900" strokeWidth={2} strokeDasharray="5,3" markerEnd="url(#arrowO)" />
                        <line x1={CX + vA.x * SCALE} y1={CY - vA.y * SCALE} x2={CX + sumX * SCALE} y2={CY - sumY * SCALE} stroke="#444" strokeWidth={1} strokeDasharray="3,3" />
                        <line x1={CX + vB.x * SCALE} y1={CY - vB.y * SCALE} x2={CX + sumX * SCALE} y2={CY - sumY * SCALE} stroke="#444" strokeWidth={1} strokeDasharray="3,3" />
                        <defs>
                          <marker id="arrowG" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#00cc88" /></marker>
                          <marker id="arrowB" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#22aaff" /></marker>
                          <marker id="arrowO" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#ff9900" /></marker>
                        </defs>
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
                }

                // ── 3D mode ──
                const a = vizVec3A, b = vizVec3B;
                const mag3 = (v: number[]) => Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
                const dot3 = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                const cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
                const sum3 = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
                const angle3 = Math.acos(Math.min(1, Math.max(-1, dot3 / (mag3(a) * mag3(b) || 1)))) * 180 / Math.PI;
                // Isometric projection: x' = (x - z) * cos30, y' = y + (x + z) * sin30
                const cos30 = Math.cos(Math.PI / 6), sin30 = 0.5;
                const proj = (v: number[]) => ({ px: CX + ((v[0] - v[2]) * cos30) * SCALE, py: CY - (v[1] + (v[0] + v[2]) * sin30) * SCALE * 0.7 });
                const pA = proj(a), pB = proj(b), pS = proj(sum3), pC = proj(cross);
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '8px', alignItems: 'center' }}>
                      <button onClick={() => setVizVector3D(false)} style={{ padding: '4px 10px', borderRadius: '4px', border: '1px solid #555', backgroundColor: 'transparent', color: '#888', fontSize: '11px', cursor: 'pointer' }}>2D</button>
                      <button onClick={() => setVizVector3D(true)} style={{ padding: '4px 10px', borderRadius: '4px', border: '1px solid #22aaff', backgroundColor: 'rgba(34,170,255,0.2)', color: '#22aaff', fontSize: '11px', cursor: 'pointer' }}>3D</button>
                    </div>
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px', flexWrap: 'wrap' }}>
                      {['x', 'y', 'z'].map((c, ci) => (
                        <label key={`a${c}`} style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#00cc88', fontSize: '11px' }}>
                          a{c}:
                          <input type="number" value={a[ci]}
                            onChange={e => { const n: [number, number, number] = [...a]; n[ci] = +e.target.value; setVizVec3A(n); }}
                            style={{ width: '45px', padding: '3px 5px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '11px', fontFamily: 'monospace' }} />
                        </label>
                      ))}
                      {['x', 'y', 'z'].map((c, ci) => (
                        <label key={`b${c}`} style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#22aaff', fontSize: '11px' }}>
                          b{c}:
                          <input type="number" value={b[ci]}
                            onChange={e => { const n: [number, number, number] = [...b]; n[ci] = +e.target.value; setVizVec3B(n); }}
                            style={{ width: '45px', padding: '3px 5px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '11px', fontFamily: 'monospace' }} />
                        </label>
                      ))}
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* 3D axis lines (isometric) */}
                      {(() => {
                        const axLen = 5;
                        const xEnd = proj([axLen, 0, 0]), yEnd = proj([0, axLen, 0]), zEnd = proj([0, 0, axLen]);
                        return (<>
                          <line x1={CX} y1={CY} x2={xEnd.px} y2={xEnd.py} stroke="#ff444488" strokeWidth={1} strokeDasharray="4,3" />
                          <text x={xEnd.px + 4} y={xEnd.py} fill="#ff4444" fontSize={10}>X</text>
                          <line x1={CX} y1={CY} x2={yEnd.px} y2={yEnd.py} stroke="#44ff4488" strokeWidth={1} strokeDasharray="4,3" />
                          <text x={yEnd.px + 4} y={yEnd.py} fill="#44ff44" fontSize={10}>Y</text>
                          <line x1={CX} y1={CY} x2={zEnd.px} y2={zEnd.py} stroke="#4444ff88" strokeWidth={1} strokeDasharray="4,3" />
                          <text x={zEnd.px + 4} y={zEnd.py} fill="#4488ff" fontSize={10}>Z</text>
                        </>);
                      })()}
                      <defs>
                        <marker id="arrowG3" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#00cc88" /></marker>
                        <marker id="arrowB3" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#22aaff" /></marker>
                        <marker id="arrowO3" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#ff9900" /></marker>
                        <marker id="arrowP3" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M0,0 L8,3 L0,6" fill="#ff66ff" /></marker>
                      </defs>
                      {/* Vector a */}
                      <line x1={CX} y1={CY} x2={pA.px} y2={pA.py} stroke="#00cc88" strokeWidth={2} markerEnd="url(#arrowG3)" />
                      <text x={pA.px + 6} y={pA.py - 6} fill="#00cc88" fontSize={12} fontWeight="bold">a⃗</text>
                      {/* Vector b */}
                      <line x1={CX} y1={CY} x2={pB.px} y2={pB.py} stroke="#22aaff" strokeWidth={2} markerEnd="url(#arrowB3)" />
                      <text x={pB.px + 6} y={pB.py - 6} fill="#22aaff" fontSize={12} fontWeight="bold">b⃗</text>
                      {/* Sum vector */}
                      <line x1={CX} y1={CY} x2={pS.px} y2={pS.py} stroke="#ff9900" strokeWidth={2} strokeDasharray="5,3" markerEnd="url(#arrowO3)" />
                      {/* Cross product vector */}
                      <line x1={CX} y1={CY} x2={pC.px} y2={pC.py} stroke="#ff66ff" strokeWidth={2} markerEnd="url(#arrowP3)" />
                      <text x={pC.px + 6} y={pC.py - 6} fill="#ff66ff" fontSize={11} fontWeight="bold">a⃗×b⃗</text>
                    </svg>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '4px 12px', marginTop: '8px', fontSize: '11px', fontFamily: 'monospace', padding: '8px', backgroundColor: '#111', borderRadius: '6px', border: '1px solid #222' }}>
                      <span style={{ color: '#00cc88' }}>|a⃗| = {mag3(a).toFixed(3)}</span>
                      <span style={{ color: '#22aaff' }}>|b⃗| = {mag3(b).toFixed(3)}</span>
                      <span style={{ color: '#ff6666' }}>a⃗·b⃗ = {dot3.toFixed(3)}</span>
                      <span style={{ color: '#b388ff' }}>θ = {angle3.toFixed(1)}°</span>
                      <span style={{ color: '#ff66ff' }}>a⃗×b⃗ = ({cross[0]}, {cross[1]}, {cross[2]})</span>
                      <span style={{ color: '#ff66ff' }}>|a⃗×b⃗| = {mag3(cross).toFixed(3)}</span>
                      <span style={{ color: '#ff9900' }}>a⃗+b⃗ = ({sum3[0]}, {sum3[1]}, {sum3[2]})</span>
                    </div>
                  </div>
                );
              })()}

              {/* Triangle visualization — right triangle with SOH CAH TOA */}
              {mathVizType === 'triangle' && (() => {
                const W = 440, H = 360, PAD = 60;
                // Right triangle: angle θ at bottom-left, right angle at bottom-right
                const angleTheta = Math.max(5, Math.min(85, vizAngle)); // reuse vizAngle for θ
                const rad = angleTheta * Math.PI / 180;
                const baseLen = 6; // adjacent
                const opp = baseLen * Math.tan(rad);
                const hyp = baseLen / Math.cos(rad);
                const maxDim = Math.max(baseLen, opp);
                const scale = (Math.min(W, H) - 2 * PAD) / maxDim;
                // Vertices: A=bottom-left (θ), B=bottom-right (90°), C=top-right
                const Ax = PAD, Ay = H - PAD;
                const Bx = PAD + baseLen * scale, By = H - PAD;
                const Cx = Bx, Cy = H - PAD - opp * scale;
                // Right angle indicator
                const sq = 12;
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '10px' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>θ =</span>
                      <input type="range" min={5} max={85} value={vizAngle} onChange={e => setVizAngle(+e.target.value)} style={{ flex: 1 }} />
                      <span style={{ color: '#fff', fontFamily: 'monospace', fontSize: '13px', minWidth: '40px' }}>{vizAngle}°</span>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* Triangle fill */}
                      <polygon points={`${Ax},${Ay} ${Bx},${By} ${Cx},${Cy}`} fill="rgba(179,136,255,0.06)" stroke="#b388ff" strokeWidth={2} />
                      {/* Right angle square at B */}
                      <polyline points={`${Bx - sq},${By} ${Bx - sq},${By - sq} ${Bx},${By - sq}`} fill="none" stroke="#888" strokeWidth={1.5} />
                      {/* θ arc at A */}
                      {(() => {
                        const arcR = 28;
                        const ex = Ax + arcR * Math.cos(-rad);
                        const ey = Ay + arcR * Math.sin(-rad);
                        return <path d={`M ${Ax + arcR},${Ay} A ${arcR} ${arcR} 0 0 0 ${ex} ${ey}`} fill="none" stroke="#ffcc00" strokeWidth={1.5} />;
                      })()}
                      <text x={Ax + 34} y={Ay - 10} fill="#ffcc00" fontSize={13} fontWeight="bold" fontFamily="monospace">θ</text>
                      {/* Side labels */}
                      {/* Adjacent (bottom) */}
                      <text x={(Ax + Bx) / 2 - 20} y={Ay + 20} fill="#00cc88" fontSize={12} fontWeight="bold" fontFamily="monospace">Adjacent</text>
                      <text x={(Ax + Bx) / 2 - 10} y={Ay + 34} fill="#00cc88" fontSize={10} fontFamily="monospace">{baseLen.toFixed(2)}</text>
                      {/* Opposite (right side) */}
                      <text x={Bx + 10} y={(By + Cy) / 2 + 4} fill="#ff6666" fontSize={12} fontWeight="bold" fontFamily="monospace">Opposite</text>
                      <text x={Bx + 10} y={(By + Cy) / 2 + 18} fill="#ff6666" fontSize={10} fontFamily="monospace">{opp.toFixed(2)}</text>
                      {/* Hypotenuse (diagonal) */}
                      {(() => {
                        const mx = (Ax + Cx) / 2, my = (Ay + Cy) / 2;
                        const ang = Math.atan2(Cy - Ay, Cx - Ax) * 180 / Math.PI;
                        return <text x={mx - 30} y={my - 10} fill="#22aaff" fontSize={12} fontWeight="bold" fontFamily="monospace" transform={`rotate(${ang} ${mx - 30} ${my - 10})`}>Hypotenuse {hyp.toFixed(2)}</text>;
                      })()}
                      {/* Vertex labels */}
                      <text x={Ax - 14} y={Ay + 6} fill="#b388ff" fontSize={11} fontWeight="bold">A</text>
                      <text x={Bx + 6} y={By + 6} fill="#b388ff" fontSize={11} fontWeight="bold">B</text>
                      <text x={Cx + 6} y={Cy - 6} fill="#b388ff" fontSize={11} fontWeight="bold">C</text>
                    </svg>
                    {/* SOH CAH TOA readout */}
                    <div style={{
                      display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', marginTop: '10px',
                      padding: '12px', borderRadius: '8px', backgroundColor: '#111', border: '1px solid #222',
                    }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ color: '#ffcc00', fontWeight: 'bold', fontSize: '13px', marginBottom: '4px' }}>SOH</div>
                        <div style={{ color: '#ff6666', fontSize: '11px', fontFamily: 'monospace' }}>sin(θ) = Opp / Hyp</div>
                        <div style={{ color: '#fff', fontSize: '13px', fontFamily: 'monospace', marginTop: '2px' }}>{Math.sin(rad).toFixed(4)}</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ color: '#ffcc00', fontWeight: 'bold', fontSize: '13px', marginBottom: '4px' }}>CAH</div>
                        <div style={{ color: '#00cc88', fontSize: '11px', fontFamily: 'monospace' }}>cos(θ) = Adj / Hyp</div>
                        <div style={{ color: '#fff', fontSize: '13px', fontFamily: 'monospace', marginTop: '2px' }}>{Math.cos(rad).toFixed(4)}</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ color: '#ffcc00', fontWeight: 'bold', fontSize: '13px', marginBottom: '4px' }}>TOA</div>
                        <div style={{ color: '#b388ff', fontSize: '11px', fontFamily: 'monospace' }}>tan(θ) = Opp / Adj</div>
                        <div style={{ color: '#fff', fontSize: '13px', fontFamily: 'monospace', marginTop: '2px' }}>{Math.tan(rad).toFixed(4)}</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: '12px', marginTop: '8px', fontSize: '11px', fontFamily: 'monospace', flexWrap: 'wrap' }}>
                      <span style={{ color: '#00cc88' }}>Adjacent = {baseLen.toFixed(2)}</span>
                      <span style={{ color: '#ff6666' }}>Opposite = {opp.toFixed(2)}</span>
                      <span style={{ color: '#22aaff' }}>Hypotenuse = {hyp.toFixed(2)}</span>
                      <span style={{ color: '#ffcc00' }}>θ = {angleTheta}°</span>
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

              {/* ── Quadratic Formula & Graph ─────────────────────────── */}
              {mathVizType === 'quadratic' && (() => {
                const a = vizQuadA, b = vizQuadB, c = vizQuadC;
                const disc = b * b - 4 * a * c;
                const vertexX = a !== 0 ? -b / (2 * a) : 0;
                const vertexY = a !== 0 ? a * vertexX * vertexX + b * vertexX + c : c;
                const roots: number[] = [];
                if (a !== 0) {
                  if (disc > 0) { roots.push((-b + Math.sqrt(disc)) / (2 * a), (-b - Math.sqrt(disc)) / (2 * a)); }
                  else if (disc === 0) { roots.push(-b / (2 * a)); }
                }
                const W = 500, H = 380, PAD = 50;
                const xSpan = Math.max(6, Math.abs(vertexX) + 5);
                const xMin = vertexX - xSpan, xMax = vertexX + xSpan;
                const pts: {x: number; y: number}[] = [];
                for (let i = 0; i <= 200; i++) {
                  const x = xMin + (xMax - xMin) * i / 200;
                  const y = a * x * x + b * x + c;
                  if (Math.abs(y) < 1e4) pts.push({ x, y });
                }
                const yVals = pts.map(p => p.y);
                const yMin = Math.min(...yVals, 0) - 1, yMax = Math.max(...yVals, 0) + 1;
                const yRange = yMax - yMin || 1, xRange = xMax - xMin || 1;
                const sx = (x: number) => PAD + ((x - xMin) / xRange) * (W - 2 * PAD);
                const sy = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - 2 * PAD);
                const toPath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '12px', marginBottom: '8px', fontSize: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#ff4444' }}>
                        a: <input type="range" min={-5} max={5} step={0.1} value={a} onChange={e => setVizQuadA(+e.target.value)} style={{ width: '80px' }} />
                        <span style={{ fontFamily: 'monospace', minWidth: '32px' }}>{a}</span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#00cc88' }}>
                        b: <input type="range" min={-10} max={10} step={0.1} value={b} onChange={e => setVizQuadB(+e.target.value)} style={{ width: '80px' }} />
                        <span style={{ fontFamily: 'monospace', minWidth: '32px' }}>{b}</span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#22aaff' }}>
                        c: <input type="range" min={-10} max={10} step={0.1} value={c} onChange={e => setVizQuadC(+e.target.value)} style={{ width: '80px' }} />
                        <span style={{ fontFamily: 'monospace', minWidth: '32px' }}>{c}</span>
                      </label>
                    </div>
                    <div style={{ fontSize: '13px', fontFamily: 'monospace', color: '#b388ff', marginBottom: '6px' }}>
                      f(x) = {a}x² {b >= 0 ? `+ ${b}` : `− ${Math.abs(b)}`}x {c >= 0 ? `+ ${c}` : `− ${Math.abs(c)}`}
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* Grid */}
                      {Array.from({ length: 11 }, (_, i) => {
                        const gx = PAD + (i / 10) * (W - 2 * PAD);
                        const gy = PAD + (i / 10) * (H - 2 * PAD);
                        return <g key={i}><line x1={gx} y1={PAD} x2={gx} y2={H - PAD} stroke="#1a1a1a" /><line x1={PAD} y1={gy} x2={W - PAD} y2={gy} stroke="#1a1a1a" /></g>;
                      })}
                      {/* Axes */}
                      {sx(0) >= PAD && sx(0) <= W - PAD && <line x1={sx(0)} y1={PAD} x2={sx(0)} y2={H - PAD} stroke="#555" strokeWidth={1.5} />}
                      {sy(0) >= PAD && sy(0) <= H - PAD && <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="#555" strokeWidth={1.5} />}
                      {/* Parabola */}
                      <path d={toPath} fill="none" stroke="#00cc88" strokeWidth={2.5} />
                      {/* Vertex */}
                      <circle cx={sx(vertexX)} cy={sy(vertexY)} r={5} fill="#ff9900" />
                      <text x={sx(vertexX) + 8} y={sy(vertexY) - 8} fill="#ff9900" fontSize={11} fontFamily="monospace">V({vertexX.toFixed(2)}, {vertexY.toFixed(2)})</text>
                      {/* Axis of symmetry */}
                      <line x1={sx(vertexX)} y1={PAD} x2={sx(vertexX)} y2={H - PAD} stroke="#ff990044" strokeWidth={1} strokeDasharray="5,4" />
                      {/* Roots */}
                      {roots.map((r, i) => (
                        <g key={i}>
                          <circle cx={sx(r)} cy={sy(0)} r={5} fill="#ff4444" />
                          <text x={sx(r) + 6} y={sy(0) - 8} fill="#ff4444" fontSize={10} fontFamily="monospace">x{roots.length > 1 ? i + 1 : ''} = {r.toFixed(3)}</text>
                        </g>
                      ))}
                      <text x={W - PAD + 5} y={sy(0) + 4} fill="#888" fontSize={11} fontWeight="bold">x</text>
                      <text x={sx(0) - 14} y={PAD - 5} fill="#888" fontSize={11} fontWeight="bold">y</text>
                    </svg>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '4px 14px', marginTop: '8px', fontSize: '11px', fontFamily: 'monospace', padding: '8px', backgroundColor: '#111', borderRadius: '6px', border: '1px solid #222' }}>
                      <span style={{ color: '#ff9900' }}>Vertex: ({vertexX.toFixed(3)}, {vertexY.toFixed(3)})</span>
                      <span style={{ color: '#b388ff' }}>Discriminant Δ = {disc.toFixed(3)}</span>
                      <span style={{ color: disc > 0 ? '#00cc88' : disc === 0 ? '#ff9900' : '#ff4444' }}>
                        {disc > 0 ? `2 real roots: ${roots.map(r => r.toFixed(3)).join(', ')}` : disc === 0 ? `1 repeated root: ${roots[0]?.toFixed(3)}` : `No real roots (Δ < 0)`}
                      </span>
                      {disc < 0 && <span style={{ color: '#888' }}>Complex: {(-b / (2 * a)).toFixed(3)} ± {(Math.sqrt(-disc) / (2 * a)).toFixed(3)}i</span>}
                      <span style={{ color: '#888' }}>Direction: {a > 0 ? '↑ opens up' : a < 0 ? '↓ opens down' : '— linear'}</span>
                      <span style={{ color: '#b388ff', gridColumn: '1 / -1' }}>Quadratic Formula: x = (−b ± √(b²−4ac)) / (2a)</span>
                    </div>
                  </div>
                );
              })()}

              {/* ── Scientific Calculator ──────────────────────────────── */}
              {mathVizType === 'calculator' && (() => {
                const symbols = [
                  { sym: 'π', ins: 'pi' }, { sym: 'e', ins: 'E' }, { sym: 'i', ins: 'I' },
                  { sym: 'sin', ins: 'sin(' }, { sym: 'cos', ins: 'cos(' }, { sym: 'tan', ins: 'tan(' },
                  { sym: 'sin⁻¹', ins: 'asin(' }, { sym: 'cos⁻¹', ins: 'acos(' }, { sym: 'tan⁻¹', ins: 'atan(' },
                  { sym: 'ln', ins: 'ln(' }, { sym: 'log', ins: 'log(' }, { sym: '√', ins: 'sqrt(' },
                  { sym: '|x|', ins: 'Abs(' }, { sym: 'x²', ins: '**2' }, { sym: 'xⁿ', ins: '**' },
                  { sym: 'n!', ins: 'factorial(' }, { sym: 'Σ', ins: 'Sum(' }, { sym: '∫', ins: 'integrate(' },
                  { sym: 'd/dx', ins: 'diff(' }, { sym: '∞', ins: 'oo' }, { sym: '(', ins: '(' }, { sym: ')', ins: ')' },
                ];
                const handleCalc = async () => {
                  if (!calcExpr.trim()) return;
                  setCalcLoading(true);
                  try {
                    const res = await fetch('http://localhost:8000/math/scientific', {
                      method: 'POST', headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ expression: calcExpr }),
                    });
                    const data = await res.json();
                    setCalcResult(data);
                    if (!data.error) {
                      setCalcHistory(prev => [{ expr: calcExpr, result: data.result || data.error || '' }, ...prev].slice(0, 20));
                    }
                  } catch { setCalcResult({ error: 'Failed to connect to server' }); }
                  setCalcLoading(false);
                };
                return (
                  <div>
                    <div style={{ fontSize: '13px', color: '#b388ff', fontWeight: 'bold', marginBottom: '8px' }}>🔬 Scientific Calculator</div>
                    {/* Symbol buttons */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '10px' }}>
                      {symbols.map(s => (
                        <button key={s.sym} onClick={() => setCalcExpr(prev => prev + s.ins)}
                          style={{ padding: '5px 10px', borderRadius: '4px', border: '1px solid #444', background: '#111', color: '#ddd', cursor: 'pointer', fontSize: '12px', fontFamily: 'monospace', minWidth: '36px', transition: 'border-color 0.2s' }}
                          onMouseEnter={e => (e.currentTarget.style.borderColor = '#b388ff')}
                          onMouseLeave={e => (e.currentTarget.style.borderColor = '#444')}>
                          {s.sym}
                        </button>
                      ))}
                    </div>
                    {/* Expression input */}
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                      <input value={calcExpr} onChange={e => setCalcExpr(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter') handleCalc(); }}
                        placeholder="e.g. integrate(sin(x), x) or diff(x**3, x) or Sum(1/n**2, (n, 1, oo))"
                        style={{ flex: 1, padding: '8px 12px', borderRadius: '8px', border: '1px solid #444', backgroundColor: '#0a0a0a', color: '#fff', fontSize: '14px', fontFamily: 'monospace' }} />
                      <button onClick={handleCalc} disabled={calcLoading}
                        style={{ padding: '8px 18px', borderRadius: '8px', border: 'none', backgroundColor: '#b388ff', color: '#000', fontWeight: 'bold', cursor: calcLoading ? 'wait' : 'pointer', fontSize: '13px' }}>
                        {calcLoading ? '...' : '='}
                      </button>
                      <button onClick={() => { setCalcExpr(''); setCalcResult(null); }}
                        style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #444', backgroundColor: 'transparent', color: '#888', cursor: 'pointer', fontSize: '13px' }}>
                        C
                      </button>
                    </div>
                    {/* Result */}
                    {calcResult && (
                      <div style={{ padding: '10px 14px', borderRadius: '8px', backgroundColor: calcResult.error ? 'rgba(255,68,68,0.08)' : 'rgba(0,204,136,0.08)', border: `1px solid ${calcResult.error ? 'rgba(255,68,68,0.2)' : 'rgba(0,204,136,0.2)'}`, marginBottom: '8px' }}>
                        {calcResult.error
                          ? <span style={{ color: '#ff6666', fontSize: '13px' }}>Error: {calcResult.error}</span>
                          : <>
                              <div style={{ fontSize: '18px', fontFamily: 'monospace', color: '#00cc88', fontWeight: 'bold' }}>{calcResult.result}</div>
                              {calcResult.result_float != null && calcResult.result_float !== calcResult.result && (
                                <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>≈ {typeof calcResult.result_float === 'number' ? calcResult.result_float.toPrecision(8) : calcResult.result_float}</div>
                              )}
                              {calcResult.latex && <div style={{ fontSize: '11px', color: '#666', marginTop: '4px', fontFamily: 'monospace' }}>LaTeX: {calcResult.latex}</div>}
                            </>
                        }
                      </div>
                    )}
                    {/* History */}
                    {calcHistory.length > 0 && (
                      <div style={{ maxHeight: '140px', overflowY: 'auto', fontSize: '11px', fontFamily: 'monospace', color: '#888' }}>
                        {calcHistory.map((h, i) => (
                          <div key={i} style={{ padding: '3px 0', borderBottom: '1px solid #1a1a1a', cursor: 'pointer' }} onClick={() => setCalcExpr(h.expr)}>
                            <span style={{ color: '#666' }}>{h.expr}</span> <span style={{ color: '#00cc88' }}>= {h.result}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* ── Derivative Animator ────────────────────────────────── */}
              {mathVizType === 'derivative' && (() => {
                const W = 500, H = 340, PAD = 45;
                const xMin = -5, xMax = 5, steps = 200;
                const safeParse = (expr: string, x: number): number => {
                  try {
                    const e = expr.replace(/\^/g, '**').replace(/sin/g, 'Math.sin').replace(/cos/g, 'Math.cos').replace(/tan/g, 'Math.tan').replace(/sqrt/g, 'Math.sqrt').replace(/abs/g, 'Math.abs').replace(/log/g, 'Math.log').replace(/exp/g, 'Math.exp').replace(/pi/g, 'Math.PI');
                    return new Function('x', `"use strict"; return (${e})`)(x) as number;
                  } catch { return NaN; }
                };
                const pts: {x: number; y: number}[] = [];
                for (let i = 0; i <= steps; i++) {
                  const x = xMin + (xMax - xMin) * i / steps;
                  const y = safeParse(vizDerivExpr, x);
                  if (isFinite(y) && Math.abs(y) < 500) pts.push({ x, y });
                }
                if (pts.length < 2) return <div style={{ color: '#666' }}>Cannot parse expression.</div>;
                const yVals = pts.map(p => p.y);
                const yMin = Math.min(...yVals) - 0.5, yMax = Math.max(...yVals) + 0.5;
                const yRange = yMax - yMin || 1, xRange = xMax - xMin;
                const sx = (x: number) => PAD + ((x - xMin) / xRange) * (W - 2 * PAD);
                const sy = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - 2 * PAD);
                const toPath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
                // Numerical derivative at vizDerivX
                const h = 0.0001;
                const fAtX = safeParse(vizDerivExpr, vizDerivX);
                const slope = (safeParse(vizDerivExpr, vizDerivX + h) - safeParse(vizDerivExpr, vizDerivX - h)) / (2 * h);
                // Tangent line endpoints
                const tLen = 2;
                const tx1 = vizDerivX - tLen, ty1 = fAtX - slope * tLen;
                const tx2 = vizDerivX + tLen, ty2 = fAtX + slope * tLen;
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '6px' }}>
                      <span style={{ color: '#00cc88', fontSize: '12px' }}>f(x) =</span>
                      <input value={vizDerivExpr} onChange={e => setVizDerivExpr(e.target.value)}
                        placeholder="x^2, sin(x), x^3 - 3*x"
                        style={{ flex: 1, padding: '6px 10px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '13px', fontFamily: 'monospace' }} />
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>x =</span>
                      <input type="range" min={-5} max={5} step={0.05} value={vizDerivX} onChange={e => setVizDerivX(+e.target.value)} style={{ flex: 1 }} />
                      <span style={{ color: '#fff', fontFamily: 'monospace', fontSize: '13px', minWidth: '40px' }}>{vizDerivX.toFixed(2)}</span>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {sx(0) >= PAD && sx(0) <= W - PAD && <line x1={sx(0)} y1={PAD} x2={sx(0)} y2={H - PAD} stroke="#555" strokeWidth={1} />}
                      {sy(0) >= PAD && sy(0) <= H - PAD && <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="#555" strokeWidth={1} />}
                      <path d={toPath} fill="none" stroke="#00cc88" strokeWidth={2} />
                      {/* Tangent line */}
                      <line x1={sx(tx1)} y1={sy(ty1)} x2={sx(tx2)} y2={sy(ty2)} stroke="#ff6666" strokeWidth={2} strokeDasharray="6,3" />
                      <circle cx={sx(vizDerivX)} cy={sy(fAtX)} r={5} fill="#ff9900" />
                      <text x={sx(vizDerivX) + 8} y={sy(fAtX) - 10} fill="#ff9900" fontSize={11} fontFamily="monospace">
                        ({vizDerivX.toFixed(2)}, {fAtX.toFixed(2)})
                      </text>
                    </svg>
                    <div style={{ display: 'flex', gap: '16px', marginTop: '6px', fontSize: '12px', fontFamily: 'monospace' }}>
                      <span style={{ color: '#00cc88' }}>f({vizDerivX.toFixed(2)}) = {fAtX.toFixed(4)}</span>
                      <span style={{ color: '#ff6666' }}>f'({vizDerivX.toFixed(2)}) = {isFinite(slope) ? slope.toFixed(4) : 'undefined'}</span>
                      <span style={{ color: '#888' }}>Tangent: y = {slope.toFixed(3)}(x − {vizDerivX.toFixed(2)}) + {fAtX.toFixed(3)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* ── Riemann Sum Builder ────────────────────────────────── */}
              {mathVizType === 'riemann' && (() => {
                const W = 500, H = 340, PAD = 45;
                const a = vizRiemannA, b_bound = vizRiemannB, n = vizRiemannN;
                const safeParse = (expr: string, x: number): number => {
                  try {
                    const e = expr.replace(/\^/g, '**').replace(/sin/g, 'Math.sin').replace(/cos/g, 'Math.cos').replace(/tan/g, 'Math.tan').replace(/sqrt/g, 'Math.sqrt').replace(/log/g, 'Math.log').replace(/exp/g, 'Math.exp').replace(/pi/g, 'Math.PI');
                    return new Function('x', `"use strict"; return (${e})`)(x) as number;
                  } catch { return NaN; }
                };
                const dx = (b_bound - a) / n;
                const rects: {x: number; y: number; w: number; h: number}[] = [];
                let areaSum = 0;
                for (let i = 0; i < n; i++) {
                  const x0 = a + i * dx;
                  const sampleX = vizRiemannMethod === 'left' ? x0 : vizRiemannMethod === 'right' ? x0 + dx : x0 + dx / 2;
                  const y = safeParse(vizRiemannExpr, sampleX);
                  if (isFinite(y)) { rects.push({ x: x0, y, w: dx, h: y }); areaSum += y * dx; }
                }
                // Curve
                const xMin = a - 0.5, xMax = b_bound + 0.5;
                const curvePts: {x: number; y: number}[] = [];
                for (let i = 0; i <= 200; i++) {
                  const x = xMin + (xMax - xMin) * i / 200;
                  const y = safeParse(vizRiemannExpr, x);
                  if (isFinite(y) && Math.abs(y) < 500) curvePts.push({ x, y });
                }
                const allY = [...curvePts.map(p => p.y), ...rects.map(r => r.h), 0];
                const yMin = Math.min(...allY) - 0.5, yMax = Math.max(...allY) + 0.5;
                const yRange = yMax - yMin || 1, xRange = xMax - xMin || 1;
                const sx = (x: number) => PAD + ((x - xMin) / xRange) * (W - 2 * PAD);
                const sy = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - 2 * PAD);
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '6px', flexWrap: 'wrap' }}>
                      <span style={{ color: '#00cc88', fontSize: '12px' }}>f(x) =</span>
                      <input value={vizRiemannExpr} onChange={e => setVizRiemannExpr(e.target.value)} style={{ width: '120px', padding: '4px 8px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px', fontFamily: 'monospace' }} />
                      <span style={{ color: '#888', fontSize: '11px' }}>a=</span>
                      <input type="number" value={a} onChange={e => setVizRiemannA(+e.target.value)} style={{ width: '50px', padding: '4px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '12px', fontFamily: 'monospace' }} />
                      <span style={{ color: '#888', fontSize: '11px' }}>b=</span>
                      <input type="number" value={b_bound} onChange={e => setVizRiemannB(+e.target.value)} style={{ width: '50px', padding: '4px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '12px', fontFamily: 'monospace' }} />
                      {(['left', 'right', 'midpoint'] as const).map(m => (
                        <button key={m} onClick={() => setVizRiemannMethod(m)} style={{ padding: '3px 8px', borderRadius: '4px', border: `1px solid ${vizRiemannMethod === m ? '#b388ff' : '#444'}`, background: vizRiemannMethod === m ? 'rgba(179,136,255,0.15)' : 'transparent', color: vizRiemannMethod === m ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '10px' }}>{m}</button>
                      ))}
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>n =</span>
                      <input type="range" min={1} max={200} value={n} onChange={e => setVizRiemannN(+e.target.value)} style={{ flex: 1 }} />
                      <span style={{ fontFamily: 'monospace', color: '#fff', fontSize: '13px', minWidth: '30px' }}>{n}</span>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {sx(0) >= PAD && sx(0) <= W - PAD && <line x1={sx(0)} y1={PAD} x2={sx(0)} y2={H - PAD} stroke="#555" strokeWidth={1} />}
                      {sy(0) >= PAD && sy(0) <= H - PAD && <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="#555" strokeWidth={1} />}
                      {/* Rectangles */}
                      {rects.map((r, i) => {
                        const rx = sx(r.x), rw = sx(r.x + r.w) - sx(r.x);
                        const ry = r.h >= 0 ? sy(r.h) : sy(0);
                        const rh = Math.abs(sy(r.h) - sy(0));
                        return <rect key={i} x={rx} y={ry} width={rw} height={rh} fill={r.h >= 0 ? 'rgba(0,204,136,0.2)' : 'rgba(255,68,68,0.2)'} stroke={r.h >= 0 ? '#00cc88' : '#ff4444'} strokeWidth={0.5} />;
                      })}
                      {/* Curve */}
                      <polyline points={curvePts.map(p => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')} fill="none" stroke="#ff9900" strokeWidth={2} />
                    </svg>
                    <div style={{ display: 'flex', gap: '16px', marginTop: '6px', fontSize: '12px', fontFamily: 'monospace' }}>
                      <span style={{ color: '#00cc88' }}>Σ ≈ {areaSum.toFixed(6)}</span>
                      <span style={{ color: '#888' }}>n = {n} rectangles, Δx = {dx.toFixed(4)}</span>
                      <span style={{ color: '#b388ff' }}>∫₍{a}₎^{'{' + b_bound + '}'} f(x)dx</span>
                    </div>
                  </div>
                );
              })()}

              {/* ── Taylor Series Expansion ────────────────────────────── */}
              {mathVizType === 'taylor' && (() => {
                const W = 500, H = 340, PAD = 45;
                const xMin = -6, xMax = 6, steps = 300;
                const fnName = vizTaylorFn;
                const nTerms = vizTaylorN;
                const fnEval: Record<string, (x: number) => number> = {
                  sin: Math.sin, cos: Math.cos, exp: Math.exp,
                };
                const fn = fnEval[fnName] || Math.sin;
                // Taylor coefficients at center=0
                const taylorCoeffs: Record<string, (n: number) => number> = {
                  sin: (n: number) => { if (n % 2 === 0) return 0; const k = (n - 1) / 2; let f = 1; for (let i = 1; i <= n; i++) f *= i; return ((-1) ** k) / f; },
                  cos: (n: number) => { if (n % 2 === 1) return 0; const k = n / 2; let f = 1; for (let i = 1; i <= n; i++) f *= i; return ((-1) ** k) / f; },
                  exp: (n: number) => { let f = 1; for (let i = 1; i <= n; i++) f *= i; return 1 / f; },
                };
                const coeff = taylorCoeffs[fnName] || taylorCoeffs.sin;
                const taylorEval = (x: number) => { let s = 0; for (let k = 0; k < nTerms; k++) { const idx = fnName === 'sin' ? 2 * k + 1 : fnName === 'cos' ? 2 * k : k; s += coeff(idx) * (x ** idx); } return s; };
                const origPts: {x: number; y: number}[] = [], approxPts: {x: number; y: number}[] = [];
                for (let i = 0; i <= steps; i++) {
                  const x = xMin + (xMax - xMin) * i / steps;
                  origPts.push({ x, y: fn(x) });
                  const ty = taylorEval(x);
                  if (isFinite(ty) && Math.abs(ty) < 20) approxPts.push({ x, y: ty });
                }
                const allY = [...origPts.map(p => p.y), ...approxPts.map(p => p.y)];
                const yMin = Math.max(Math.min(...allY), -8) - 0.5, yMax = Math.min(Math.max(...allY), 8) + 0.5;
                const yRange = yMax - yMin || 1, xRange = xMax - xMin;
                const sx = (x: number) => PAD + ((x - xMin) / xRange) * (W - 2 * PAD);
                const sy = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - 2 * PAD);
                const origPath = origPts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
                const approxPath = approxPts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>Function:</span>
                      {(['sin', 'cos', 'exp'] as const).map(f => (
                        <button key={f} onClick={() => setVizTaylorFn(f)} style={{ padding: '3px 10px', borderRadius: '4px', border: `1px solid ${vizTaylorFn === f ? '#b388ff' : '#444'}`, background: vizTaylorFn === f ? 'rgba(179,136,255,0.15)' : 'transparent', color: vizTaylorFn === f ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '12px' }}>{f}(x)</button>
                      ))}
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ color: '#ff9900', fontSize: '12px' }}>Terms:</span>
                      <input type="range" min={1} max={15} value={nTerms} onChange={e => setVizTaylorN(+e.target.value)} style={{ flex: 1 }} />
                      <span style={{ fontFamily: 'monospace', color: '#fff', fontSize: '13px', minWidth: '20px' }}>{nTerms}</span>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {sx(0) >= PAD && sx(0) <= W - PAD && <line x1={sx(0)} y1={PAD} x2={sx(0)} y2={H - PAD} stroke="#555" strokeWidth={1} />}
                      {sy(0) >= PAD && sy(0) <= H - PAD && <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="#555" strokeWidth={1} />}
                      <path d={origPath} fill="none" stroke="#00cc88" strokeWidth={2} />
                      <path d={approxPath} fill="none" stroke="#ff9900" strokeWidth={2} strokeDasharray="6,3" />
                    </svg>
                    <div style={{ display: 'flex', gap: '16px', marginTop: '6px', fontSize: '11px' }}>
                      <span style={{ color: '#00cc88' }}>— {fnName}(x)</span>
                      <span style={{ color: '#ff9900' }}>-- Taylor ({nTerms} terms)</span>
                    </div>
                  </div>
                );
              })()}

              {/* ── Fourier Series ─────────────────────────────────────── */}
              {mathVizType === 'fourier' && (() => {
                const W = 500, H = 300, PAD = 40;
                const nHarm = vizFourierN;
                const steps = 400;
                const xMin = -Math.PI, xMax = 3 * Math.PI;
                const fourierEval: Record<string, (x: number, n: number) => number> = {
                  square: (x, n) => { let s = 0; for (let k = 1; k <= n; k++) s += Math.sin((2 * k - 1) * x) / (2 * k - 1); return (4 / Math.PI) * s; },
                  sawtooth: (x, n) => { let s = 0; for (let k = 1; k <= n; k++) s += ((-1) ** (k + 1)) * Math.sin(k * x) / k; return (2 / Math.PI) * s; },
                  triangle: (x, n) => { let s = 0; for (let k = 0; k < n; k++) { const m = 2 * k + 1; s += ((-1) ** k) * Math.sin(m * x) / (m * m); } return (8 / (Math.PI * Math.PI)) * s; },
                };
                const evalFn = fourierEval[vizFourierWave] || fourierEval.square;
                const targetFn: Record<string, (x: number) => number> = {
                  square: (x) => { const p = ((x % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI); return p < Math.PI ? 1 : -1; },
                  sawtooth: (x) => { const p = ((x % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI); return (p / Math.PI) - 1; },
                  triangle: (x) => { const p = ((x % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI); return p < Math.PI ? (2 * p / Math.PI - 1) : (3 - 2 * p / Math.PI); },
                };
                const target = targetFn[vizFourierWave] || targetFn.square;
                const targPts: string[] = [], approxPts: string[] = [];
                let yMin = -1.5, yMax = 1.5;
                for (let i = 0; i <= steps; i++) {
                  const x = xMin + (xMax - xMin) * i / steps;
                  const ty = target(x), ay = evalFn(x, nHarm);
                  if (Math.abs(ty) < 5) { targPts.push(`${x},${ty}`); } 
                  if (Math.abs(ay) < 5) { approxPts.push(`${x},${ay}`); if (ay > yMax) yMax = ay; if (ay < yMin) yMin = ay; }
                }
                const xRange = xMax - xMin, yRange = yMax - yMin || 1;
                const sx = (x: number) => PAD + ((x - xMin) / xRange) * (W - 2 * PAD);
                const sy = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - 2 * PAD);
                const toSvg = (pts: string[]) => pts.map((p, i) => { const [x, y] = p.split(',').map(Number); return `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`; }).join(' ');
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>Waveform:</span>
                      {(['square', 'sawtooth', 'triangle'] as const).map(w => (
                        <button key={w} onClick={() => setVizFourierWave(w)} style={{ padding: '3px 10px', borderRadius: '4px', border: `1px solid ${vizFourierWave === w ? '#b388ff' : '#444'}`, background: vizFourierWave === w ? 'rgba(179,136,255,0.15)' : 'transparent', color: vizFourierWave === w ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '12px' }}>{w}</button>
                      ))}
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ color: '#ff9900', fontSize: '12px' }}>Harmonics:</span>
                      <input type="range" min={1} max={50} value={nHarm} onChange={e => setVizFourierN(+e.target.value)} style={{ flex: 1 }} />
                      <span style={{ fontFamily: 'monospace', color: '#fff', fontSize: '13px', minWidth: '24px' }}>{nHarm}</span>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {sy(0) >= PAD && sy(0) <= H - PAD && <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="#555" strokeWidth={1} />}
                      <path d={toSvg(targPts)} fill="none" stroke="#555" strokeWidth={1.5} strokeDasharray="4,3" />
                      <path d={toSvg(approxPts)} fill="none" stroke="#00cc88" strokeWidth={2} />
                    </svg>
                    <div style={{ display: 'flex', gap: '16px', marginTop: '6px', fontSize: '11px' }}>
                      <span style={{ color: '#555' }}>-- target ({vizFourierWave})</span>
                      <span style={{ color: '#00cc88' }}>— Fourier ({nHarm} harmonics)</span>
                    </div>
                  </div>
                );
              })()}

              {/* ── Parametric Curves ──────────────────────────────────── */}
              {mathVizType === 'parametric' && (() => {
                const W = 400, H = 400, PAD = 40;
                const safeParse = (expr: string, t: number): number => {
                  try {
                    const e = expr.replace(/\^/g, '**').replace(/sin/g, 'Math.sin').replace(/cos/g, 'Math.cos').replace(/tan/g, 'Math.tan').replace(/sqrt/g, 'Math.sqrt').replace(/pi/g, 'Math.PI');
                    return new Function('t', `"use strict"; return (${e})`)(t) as number;
                  } catch { return NaN; }
                };
                const pts: {x: number; y: number}[] = [];
                for (let i = 0; i <= 300; i++) {
                  const t = (2 * Math.PI) * i / 300;
                  const px = safeParse(vizParamXExpr, t), py = safeParse(vizParamYExpr, t);
                  if (isFinite(px) && isFinite(py) && Math.abs(px) < 100 && Math.abs(py) < 100) pts.push({ x: px, y: py });
                }
                const curPx = safeParse(vizParamXExpr, vizParamT), curPy = safeParse(vizParamYExpr, vizParamT);
                if (pts.length < 2) return <div style={{ color: '#666' }}>Cannot parse parametric expressions.</div>;
                const xVals = pts.map(p => p.x), yVals = pts.map(p => p.y);
                const xMin = Math.min(...xVals) - 0.3, xMax = Math.max(...xVals) + 0.3;
                const yMin = Math.min(...yVals) - 0.3, yMax = Math.max(...yVals) + 0.3;
                const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
                const sx = (x: number) => PAD + ((x - xMin) / xRange) * (W - 2 * PAD);
                const sy = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - 2 * PAD);
                const curvePath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '6px', flexWrap: 'wrap' }}>
                      <span style={{ color: '#00cc88', fontSize: '12px' }}>x(t) =</span>
                      <input value={vizParamXExpr} onChange={e => setVizParamXExpr(e.target.value)} style={{ width: '130px', padding: '4px 8px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px', fontFamily: 'monospace' }} />
                      <span style={{ color: '#22aaff', fontSize: '12px' }}>y(t) =</span>
                      <input value={vizParamYExpr} onChange={e => setVizParamYExpr(e.target.value)} style={{ width: '130px', padding: '4px 8px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px', fontFamily: 'monospace' }} />
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>t =</span>
                      <input type="range" min={0} max={6.28} step={0.02} value={vizParamT} onChange={e => setVizParamT(+e.target.value)} style={{ flex: 1 }} />
                      <span style={{ fontFamily: 'monospace', color: '#fff', fontSize: '13px', minWidth: '36px' }}>{vizParamT.toFixed(2)}</span>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {sx(0) >= PAD && sx(0) <= W - PAD && <line x1={sx(0)} y1={PAD} x2={sx(0)} y2={H - PAD} stroke="#333" strokeWidth={1} />}
                      {sy(0) >= PAD && sy(0) <= H - PAD && <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="#333" strokeWidth={1} />}
                      <path d={curvePath} fill="none" stroke="#00cc88" strokeWidth={2} />
                      {isFinite(curPx) && isFinite(curPy) && <circle cx={sx(curPx)} cy={sy(curPy)} r={5} fill="#ff4444" />}
                    </svg>
                    <div style={{ fontSize: '11px', fontFamily: 'monospace', color: '#ff4444', marginTop: '6px' }}>
                      P(t={vizParamT.toFixed(2)}) = ({isFinite(curPx) ? curPx.toFixed(3) : '?'}, {isFinite(curPy) ? curPy.toFixed(3) : '?'})
                    </div>
                    <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>Try: x(t) = 2*cos(t), y(t) = sin(2*t) for a Lissajous figure</div>
                  </div>
                );
              })()}

              {/* ── Regression / Correlation ───────────────────────────── */}
              {mathVizType === 'regression' && (() => {
                const W = 440, H = 340, PAD = 45;
                const points = vizRegPoints;
                if (points.length < 2) return <div style={{ color: '#666' }}>Enter at least 2 points.</div>;
                const n = points.length;
                const sx_sum = points.reduce((s, p) => s + p.x, 0);
                const sy_sum = points.reduce((s, p) => s + p.y, 0);
                const sxy = points.reduce((s, p) => s + p.x * p.y, 0);
                const sx2 = points.reduce((s, p) => s + p.x * p.x, 0);
                const xMean = sx_sum / n, yMean = sy_sum / n;
                const slope = (n * sxy - sx_sum * sy_sum) / (n * sx2 - sx_sum * sx_sum);
                const intercept = yMean - slope * xMean;
                // R²
                const ssTot = points.reduce((s, p) => s + (p.y - yMean) ** 2, 0);
                const ssRes = points.reduce((s, p) => s + (p.y - (slope * p.x + intercept)) ** 2, 0);
                const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
                const xVals = points.map(p => p.x), yVals = points.map(p => p.y);
                const xMin = Math.min(...xVals) - 1, xMax = Math.max(...xVals) + 1;
                const yMin = Math.min(...yVals, slope * xMin + intercept, slope * xMax + intercept) - 1;
                const yMax = Math.max(...yVals, slope * xMin + intercept, slope * xMax + intercept) + 1;
                const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
                const toX = (x: number) => PAD + ((x - xMin) / xRange) * (W - 2 * PAD);
                const toY = (y: number) => H - PAD - ((y - yMin) / yRange) * (H - 2 * PAD);
                return (
                  <div>
                    {/* Point editor */}
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                      {points.map((p, i) => (
                        <span key={i} style={{ display: 'flex', gap: '2px', fontSize: '10px', alignItems: 'center' }}>
                          <span style={{ color: '#888' }}>(</span>
                          <input type="number" value={p.x} onChange={e => { const n = [...points]; n[i] = { ...n[i], x: +e.target.value }; setVizRegPoints(n); }} style={{ width: '36px', padding: '2px', borderRadius: '3px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '10px', fontFamily: 'monospace' }} />
                          <span style={{ color: '#888' }}>,</span>
                          <input type="number" value={p.y} onChange={e => { const n = [...points]; n[i] = { ...n[i], y: +e.target.value }; setVizRegPoints(n); }} style={{ width: '36px', padding: '2px', borderRadius: '3px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '10px', fontFamily: 'monospace' }} />
                          <span style={{ color: '#888' }}>)</span>
                          <button onClick={() => setVizRegPoints(points.filter((_, j) => j !== i))} style={{ background: 'none', border: 'none', color: '#ff4444', cursor: 'pointer', fontSize: '10px', padding: '0 2px' }}>×</button>
                        </span>
                      ))}
                      <button onClick={() => setVizRegPoints([...points, { x: Math.round(Math.random() * 10), y: Math.round(Math.random() * 10) }])} style={{ padding: '2px 8px', borderRadius: '4px', border: '1px solid #444', background: 'transparent', color: '#00cc88', cursor: 'pointer', fontSize: '10px' }}>+ Add</button>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '10px', color: '#888', cursor: 'pointer', marginLeft: '8px' }}>
                        <input type="checkbox" checked={vizRegShowResiduals} onChange={e => setVizRegShowResiduals(e.target.checked)} style={{ accentColor: '#ff9900' }} /> Residuals
                      </label>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {toX(0) >= PAD && toX(0) <= W - PAD && <line x1={toX(0)} y1={PAD} x2={toX(0)} y2={H - PAD} stroke="#333" />}
                      {toY(0) >= PAD && toY(0) <= H - PAD && <line x1={PAD} y1={toY(0)} x2={W - PAD} y2={toY(0)} stroke="#333" />}
                      {/* Best-fit line */}
                      <line x1={toX(xMin)} y1={toY(slope * xMin + intercept)} x2={toX(xMax)} y2={toY(slope * xMax + intercept)} stroke="#ff9900" strokeWidth={2} />
                      {/* Residuals */}
                      {vizRegShowResiduals && points.map((p, i) => {
                        const predicted = slope * p.x + intercept;
                        return <line key={i} x1={toX(p.x)} y1={toY(p.y)} x2={toX(p.x)} y2={toY(predicted)} stroke="#ff444488" strokeWidth={1} strokeDasharray="3,2" />;
                      })}
                      {/* Points */}
                      {points.map((p, i) => <circle key={i} cx={toX(p.x)} cy={toY(p.y)} r={5} fill="#22aaff" />)}
                    </svg>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '4px 14px', marginTop: '6px', fontSize: '11px', fontFamily: 'monospace', padding: '6px 8px', backgroundColor: '#111', borderRadius: '6px', border: '1px solid #222' }}>
                      <span style={{ color: '#ff9900' }}>y = {slope.toFixed(3)}x + {intercept.toFixed(3)}</span>
                      <span style={{ color: '#22aaff' }}>R² = {r2.toFixed(4)}</span>
                      <span style={{ color: '#888' }}>n = {n} points</span>
                    </div>
                  </div>
                );
              })()}

              {/* ── Monte Carlo Pi Estimator ───────────────────────────── */}
              {mathVizType === 'montecarlo' && (() => {
                // Generate points on demand
                if (vizMontePoints.length !== vizMonteN) {
                  const pts: {x: number; y: number; inside: boolean}[] = [];
                  for (let i = 0; i < vizMonteN; i++) {
                    const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
                    pts.push({ x, y, inside: x * x + y * y <= 1 });
                  }
                  // Use setTimeout to avoid setState during render
                  setTimeout(() => setVizMontePoints(pts), 0);
                }
                const pts = vizMontePoints;
                const inside = pts.filter(p => p.inside).length;
                const piEst = pts.length > 0 ? 4 * inside / pts.length : 0;
                const W = 360, H = 360, CX = W / 2, CY = H / 2, R = 160;
                const sx = (x: number) => CX + x * R;
                const sy = (y: number) => CY - y * R;
                return (
                  <div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ color: '#b388ff', fontSize: '12px' }}>Points:</span>
                      <input type="range" min={10} max={5000} step={10} value={vizMonteN} onChange={e => { setVizMonteN(+e.target.value); setVizMontePoints([]); }} style={{ flex: 1 }} />
                      <span style={{ fontFamily: 'monospace', color: '#fff', fontSize: '13px', minWidth: '40px' }}>{vizMonteN}</span>
                      <button onClick={() => setVizMontePoints([])} style={{ padding: '4px 10px', borderRadius: '4px', border: '1px solid #444', background: 'transparent', color: '#00cc88', cursor: 'pointer', fontSize: '11px' }}>Re-roll</button>
                    </div>
                    <svg width={W} height={H} style={{ backgroundColor: '#0a0a0a', borderRadius: '8px', border: '1px solid #222' }}>
                      {/* Square boundary */}
                      <rect x={CX - R} y={CY - R} width={R * 2} height={R * 2} fill="none" stroke="#333" strokeWidth={1} />
                      {/* Circle */}
                      <circle cx={CX} cy={CY} r={R} fill="none" stroke="#555" strokeWidth={1.5} />
                      {/* Points */}
                      {pts.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={1.5} fill={p.inside ? '#00cc88' : '#ff4444'} opacity={0.7} />)}
                    </svg>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '4px 14px', marginTop: '6px', fontSize: '12px', fontFamily: 'monospace', padding: '8px', backgroundColor: '#111', borderRadius: '6px', border: '1px solid #222' }}>
                      <span style={{ color: '#00cc88', fontWeight: 'bold', fontSize: '16px' }}>π ≈ {piEst.toFixed(6)}</span>
                      <span style={{ color: '#888' }}>Actual: {Math.PI.toFixed(6)}</span>
                      <span style={{ color: '#888' }}>Error: {Math.abs(piEst - Math.PI).toFixed(6)}</span>
                      <span style={{ color: '#00cc88' }}>Inside: {inside}</span>
                      <span style={{ color: '#ff4444' }}>Outside: {pts.length - inside}</span>
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

      {/* ── Image Generation Page ──────────────────────────────────── */}
      {activeMode === 'imagegen' && (
        <div style={{
          position: 'fixed', top: '60px', left: '16px', right: '16px',
          bottom: '30px',
          display: 'flex', gap: '16px', color: '#fff', zIndex: 30,
        }}>
          {/* Left panel: Settings */}
          <div style={{
            width: '300px', flexShrink: 0,
            backgroundColor: 'rgba(10,10,20,0.95)', borderRadius: '14px',
            border: '1px solid rgba(179,136,255,0.2)', backdropFilter: 'blur(16px)',
            overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px',
          }}>
            <h3 style={{ margin: 0, fontSize: '15px', color: '#b388ff', display: 'flex', alignItems: 'center', gap: '8px' }}>🎨 ImageStudio</h3>

            {/* Sub-tab selector */}
            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
              {([['generate','🖼'], ['animate','🎬'], ['storyboard','📋'], ['video','🎥'], ['upscale','🔍'], ['train','🧪']] as const).map(([tab, icon]) => (
                <button key={tab} onClick={() => setImageGenSubTab(tab as any)} style={{
                  flex: 1, minWidth: '50px', padding: '5px 4px', borderRadius: '6px', border: '1px solid',
                  borderColor: imageGenSubTab === tab ? '#b388ff' : '#333',
                  backgroundColor: imageGenSubTab === tab ? 'rgba(179,136,255,0.15)' : '#111',
                  color: imageGenSubTab === tab ? '#b388ff' : '#666',
                  cursor: 'pointer', fontSize: '10px', textTransform: 'capitalize',
                }}>{icon} {tab}</button>
              ))}
            </div>

            {/* Model selector (shared) */}
            <div>
              <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>Checkpoint Model</label>
              <select
                value={selectedImageModel}
                onChange={e => setSelectedImageModel(e.target.value)}
                style={{ width: '100%', padding: '7px 10px', borderRadius: '8px', backgroundColor: '#111', color: '#fff', border: '1px solid #333', fontSize: '12px' }}
              >
                <option value="">Auto (first available)</option>
                {imageModels.filter(m => m.type === 'checkpoint').map(m => (
                  <option key={m.name} value={m.name}>{m.name} ({m.size_mb}MB)</option>
                ))}
              </select>
            </div>

            {/* Character / LoRA Search (always visible) */}
            <div>
              <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>🔍 Search Characters & LoRAs</label>
              <input type="text" value={loraSearchQuery}
                onChange={e => { setLoraSearchQuery(e.target.value); searchLoras(e.target.value); }}
                placeholder="Search by name, trigger word, or past prompt..."
                style={{ width: '100%', padding: '7px 10px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px', outline: 'none', boxSizing: 'border-box' }} />

              {/* Search results dropdown */}
              {loraSearchQuery && (loraSearchResults.length > 0 || loraSearchHistoryResults.length > 0) && (
                <div style={{
                  border: '1px solid #333', borderRadius: '8px', backgroundColor: '#0a0a0a',
                  padding: '8px', maxHeight: '240px', overflowY: 'auto', marginTop: '4px',
                }}>
                  {/* LoRA matches */}
                  {loraSearchResults.length > 0 && (
                    <div style={{ marginBottom: loraSearchHistoryResults.length > 0 ? '8px' : '0' }}>
                      <div style={{ fontSize: '10px', color: '#b388ff', marginBottom: '4px', fontWeight: 'bold' }}>LoRA Matches</div>
                      {loraSearchResults.map((lr: any) => (
                        <div key={lr.name} style={{ padding: '4px 6px', fontSize: '11px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderRadius: '4px', backgroundColor: '#111', marginBottom: '2px' }}>
                          <div>
                            <span style={{ color: '#ccc' }}>{lr.name}</span>
                            <span style={{ color: '#555', fontSize: '9px', marginLeft: '6px' }}>{lr.category}</span>
                            {lr.trigger_words?.length > 0 && (
                              <div style={{ marginTop: '2px' }}>
                                {lr.trigger_words.map((tw: string, j: number) => (
                                  <span key={j} onClick={() => insertTriggerWord(tw)}
                                    style={{ fontSize: '9px', color: '#00cc88', cursor: 'pointer', marginRight: '4px', textDecoration: 'underline' }}>{tw}</span>
                                ))}
                              </div>
                            )}
                          </div>
                          <button onClick={() => setSelectedLoras(prev => prev.includes(lr.name) ? prev.filter(x => x !== lr.name) : [...prev, lr.name])}
                            style={{ background: 'none', border: '1px solid', borderColor: selectedLoras.includes(lr.name) ? '#b388ff' : '#444', borderRadius: '4px', color: selectedLoras.includes(lr.name) ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '9px', padding: '1px 6px' }}>
                            {selectedLoras.includes(lr.name) ? '✓' : '+ Use'}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* History matches */}
                  {loraSearchHistoryResults.length > 0 && (
                    <div>
                      <div style={{ fontSize: '10px', color: '#ff9900', marginBottom: '4px', fontWeight: 'bold', borderTop: loraSearchResults.length > 0 ? '1px solid #222' : 'none', paddingTop: loraSearchResults.length > 0 ? '6px' : '0' }}>Past Generations</div>
                      {loraSearchHistoryResults.map((h: any, i: number) => (
                        <div key={i} onClick={() => {
                          if (h.prompt) setImageGenPrompt(h.prompt);
                          if (h.seed && h.seed !== -1) setImageGenSeed(h.seed);
                        }} style={{
                          padding: '4px 6px', fontSize: '11px', borderRadius: '4px', backgroundColor: '#111',
                          marginBottom: '2px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        }}>
                          <div style={{ flex: 1, overflow: 'hidden' }}>
                            <span style={{ color: '#ccc', fontSize: '10px' }}>{h.prompt || 'Unknown'}</span>
                            {h.timestamp && <div style={{ color: '#444', fontSize: '9px', marginTop: '1px' }}>{new Date(h.timestamp).toLocaleString()}</div>}
                          </div>
                          {h.seed && <span style={{ color: '#555', fontSize: '9px', marginLeft: '6px', flexShrink: 0 }}>seed:{h.seed}</span>}
                        </div>
                      ))}
                    </div>
                  )}

                  {loraSearchResults.length === 0 && loraSearchHistoryResults.length === 0 && (
                    <div style={{ color: '#444', fontSize: '10px', textAlign: 'center', padding: '8px 0' }}>No matches found</div>
                  )}
                </div>
              )}
            </div>

            {/* LoRA Browser (categorized with trigger words) */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <label style={{ fontSize: '11px', color: '#888' }}>LoRA Adapters</label>
                <button onClick={() => { setLoraBrowserOpen(!loraBrowserOpen); if (!loraBrowserOpen) fetchLoraCategories(); }}
                  style={{ background: 'none', border: '1px solid #444', borderRadius: '4px', color: '#b388ff', cursor: 'pointer', fontSize: '10px', padding: '2px 8px' }}>
                  {loraBrowserOpen ? '▼ Close' : '▶ Browse'}
                </button>
              </div>

              {/* Selected LoRAs chips */}
              {selectedLoras.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '6px' }}>
                  {selectedLoras.map(l => (
                    <span key={l} style={{
                      padding: '2px 8px', borderRadius: '10px', fontSize: '10px',
                      backgroundColor: 'rgba(179,136,255,0.15)', color: '#b388ff', border: '1px solid rgba(179,136,255,0.3)',
                      display: 'flex', alignItems: 'center', gap: '4px',
                    }}>
                      {l}
                      <span onClick={() => setSelectedLoras(prev => prev.filter(x => x !== l))} style={{ cursor: 'pointer', color: '#ff4444', fontWeight: 'bold' }}>×</span>
                    </span>
                  ))}
                </div>
              )}

              {/* LoRA Browser panel */}
              {loraBrowserOpen && (
                <div style={{
                  border: '1px solid #333', borderRadius: '8px', backgroundColor: '#0a0a0a',
                  padding: '8px', maxHeight: '320px', overflowY: 'auto',
                }}
                  onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); (e.currentTarget as HTMLElement).style.borderColor = '#b388ff'; }}
                  onDragLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = '#333'; }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    (e.currentTarget as HTMLElement).style.borderColor = '#333';
                    const files = e.dataTransfer?.files;
                    if (!files || files.length === 0) return;
                    const file = files[0];
                    if (!file.name.endsWith('.safetensors')) return;
                    // Extract LoRA name (strip extension)
                    const loraName = file.name.replace('.safetensors', '');
                    // Auto-open trigger word editor for this LoRA
                    setEditingTriggerLora(loraName);
                    setTriggerWordInput('');
                    // Try to find existing trigger words
                    for (const cat of Object.values(loraCategories)) {
                      const match = (cat as any[]).find((lr: any) => lr.name === loraName);
                      if (match?.trigger_words?.length > 0) {
                        setTriggerWordInput(match.trigger_words.join(', '));
                        break;
                      }
                    }
                  }}
                >
                  {/* Search */}
                  <input type="text" value={loraSearchQuery}
                    onChange={e => { setLoraSearchQuery(e.target.value); searchLoras(e.target.value); }}
                    placeholder="Search LoRAs by name or trigger word..."
                    style={{ width: '100%', padding: '5px 8px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '11px', marginBottom: '6px', outline: 'none', boxSizing: 'border-box' }} />

                  {/* Drag-drop hint */}
                  <div style={{ fontSize: '9px', color: '#555', textAlign: 'center', marginBottom: '4px' }}>
                    Drop a .safetensors file here to add trigger words
                  </div>

                  {/* Floating trigger word editor for dropped LoRAs not in any category */}
                  {editingTriggerLora && !Object.values(loraCategories).flat().some((lr: any) => lr.name === editingTriggerLora) && (
                    <div style={{ padding: '8px', borderRadius: '6px', backgroundColor: '#1a1a2e', border: '1px solid #b388ff', marginBottom: '8px' }}>
                      <div style={{ fontSize: '11px', color: '#b388ff', marginBottom: '4px', fontWeight: 'bold' }}>✏️ Set trigger words for: {editingTriggerLora}</div>
                      <div style={{ display: 'flex', gap: '4px' }}>
                        <input type="text" value={triggerWordInput}
                          onChange={e => setTriggerWordInput(e.target.value)}
                          onKeyDown={e => { if (e.key === 'Enter') { saveTriggerWords(editingTriggerLora!, triggerWordInput); fetchLoraCategories(); } }}
                          placeholder="word1, word2, ..."
                          autoFocus
                          style={{ flex: 1, padding: '4px 6px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#0a0a0a', color: '#fff', fontSize: '10px', outline: 'none' }} />
                        <button onClick={() => { saveTriggerWords(editingTriggerLora!, triggerWordInput); fetchLoraCategories(); }}
                          style={{ background: '#00cc88', border: 'none', borderRadius: '4px', color: '#000', cursor: 'pointer', fontSize: '9px', padding: '4px 8px', fontWeight: 'bold' }}>Save</button>
                        <button onClick={() => { setEditingTriggerLora(null); setTriggerWordInput(''); }}
                          style={{ background: 'none', border: '1px solid #444', borderRadius: '4px', color: '#888', cursor: 'pointer', fontSize: '9px', padding: '4px 6px' }}>✕</button>
                      </div>
                    </div>
                  )}

                  {/* Search results */}
                  {loraSearchQuery && loraSearchResults.length > 0 && (
                    <div style={{ marginBottom: '8px', borderBottom: '1px solid #222', paddingBottom: '6px' }}>
                      <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px' }}>Search Results</div>
                      {loraSearchResults.map((lr: any) => (
                        <div key={lr.name} style={{ padding: '4px 6px', fontSize: '11px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderRadius: '4px', backgroundColor: '#111', marginBottom: '2px' }}>
                          <div>
                            <span style={{ color: '#ccc' }}>{lr.name}</span>
                            <span style={{ color: '#555', fontSize: '9px', marginLeft: '6px' }}>{lr.category}</span>
                            {lr.trigger_words?.length > 0 && (
                              <div style={{ marginTop: '2px' }}>
                                {lr.trigger_words.map((tw: string, j: number) => (
                                  <span key={j} onClick={() => insertTriggerWord(tw)}
                                    style={{ fontSize: '9px', color: '#00cc88', cursor: 'pointer', marginRight: '4px', textDecoration: 'underline' }}>{tw}</span>
                                ))}
                              </div>
                            )}
                          </div>
                          <button onClick={() => setSelectedLoras(prev => prev.includes(lr.name) ? prev.filter(x => x !== lr.name) : [...prev, lr.name])}
                            style={{ background: 'none', border: '1px solid', borderColor: selectedLoras.includes(lr.name) ? '#b388ff' : '#444', borderRadius: '4px', color: selectedLoras.includes(lr.name) ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '9px', padding: '1px 6px' }}>
                            {selectedLoras.includes(lr.name) ? '✓' : '+'}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Category tabs */}
                  <div style={{ display: 'flex', gap: '2px', flexWrap: 'wrap', marginBottom: '6px' }}>
                    {['styles', 'characters', 'clothing', 'poses', 'concept', 'action'].map(cat => {
                      const count = (loraCategories[cat] || []).length;
                      return (
                        <button key={cat} onClick={() => setLoraBrowserCategory(cat)} style={{
                          padding: '3px 6px', borderRadius: '4px', fontSize: '9px', cursor: 'pointer',
                          border: '1px solid', textTransform: 'capitalize',
                          borderColor: loraBrowserCategory === cat ? '#b388ff' : '#333',
                          backgroundColor: loraBrowserCategory === cat ? 'rgba(179,136,255,0.15)' : 'transparent',
                          color: loraBrowserCategory === cat ? '#b388ff' : '#666',
                        }}>{cat} ({count})</button>
                      );
                    })}
                  </div>

                  {/* LoRAs in selected category */}
                  {(loraCategories[loraBrowserCategory] || []).length === 0 && (
                    <div style={{ color: '#444', fontSize: '10px', textAlign: 'center', padding: '12px 0' }}>No LoRAs in {loraBrowserCategory}/</div>
                  )}
                  {(loraCategories[loraBrowserCategory] || []).map(lora => (
                    <div key={lora.name} style={{
                      padding: '6px', borderRadius: '6px', backgroundColor: '#111', border: '1px solid #222',
                      marginBottom: '4px', fontSize: '11px',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ color: '#ccc', fontWeight: 500 }}>{lora.name}</span>
                        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                          <span style={{ color: '#555', fontSize: '9px' }}>{lora.size_mb}MB</span>
                          <button onClick={() => setSelectedLoras(prev => prev.includes(lora.name) ? prev.filter(x => x !== lora.name) : [...prev, lora.name])}
                            style={{ background: 'none', border: '1px solid', borderColor: selectedLoras.includes(lora.name) ? '#b388ff' : '#444', borderRadius: '4px', color: selectedLoras.includes(lora.name) ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '9px', padding: '1px 6px' }}>
                            {selectedLoras.includes(lora.name) ? '✓ Active' : '+ Use'}
                          </button>
                        </div>
                      </div>

                      {/* Trigger words display */}
                      {lora.trigger_words?.length > 0 && editingTriggerLora !== lora.name && (
                        <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '3px', alignItems: 'center' }}>
                          <span style={{ fontSize: '9px', color: '#666' }}>Triggers:</span>
                          {lora.trigger_words.map((tw: string, j: number) => (
                            <span key={j} onClick={() => insertTriggerWord(tw)}
                              style={{
                                padding: '1px 6px', borderRadius: '8px', fontSize: '9px',
                                backgroundColor: 'rgba(0,204,136,0.1)', color: '#00cc88',
                                border: '1px solid rgba(0,204,136,0.2)', cursor: 'pointer',
                              }} title="Click to insert into prompt">{tw}</span>
                          ))}
                          <span onClick={() => { setEditingTriggerLora(lora.name); setTriggerWordInput(lora.trigger_words.join(', ')); }}
                            style={{ fontSize: '9px', color: '#555', cursor: 'pointer' }} title="Edit triggers">✏️</span>
                        </div>
                      )}

                      {/* No trigger words — add button */}
                      {(!lora.trigger_words || lora.trigger_words.length === 0) && editingTriggerLora !== lora.name && (
                        <button onClick={() => { setEditingTriggerLora(lora.name); setTriggerWordInput(''); }}
                          style={{ marginTop: '4px', background: 'none', border: '1px dashed #444', borderRadius: '4px', color: '#666', cursor: 'pointer', fontSize: '9px', padding: '2px 8px', width: '100%' }}>
                          + Add trigger words
                        </button>
                      )}

                      {/* Trigger word edit mode */}
                      {editingTriggerLora === lora.name && (
                        <div style={{ marginTop: '4px', display: 'flex', gap: '4px' }}>
                          <input type="text" value={triggerWordInput}
                            onChange={e => setTriggerWordInput(e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter') saveTriggerWords(lora.name, triggerWordInput); }}
                            placeholder="word1, word2, ..."
                            style={{ flex: 1, padding: '3px 6px', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#0a0a0a', color: '#fff', fontSize: '10px', outline: 'none' }} />
                          <button onClick={() => saveTriggerWords(lora.name, triggerWordInput)}
                            style={{ background: '#00cc88', border: 'none', borderRadius: '4px', color: '#000', cursor: 'pointer', fontSize: '9px', padding: '3px 8px', fontWeight: 'bold' }}>Save</button>
                          <button onClick={() => { setEditingTriggerLora(null); setTriggerWordInput(''); }}
                            style={{ background: 'none', border: '1px solid #444', borderRadius: '4px', color: '#888', cursor: 'pointer', fontSize: '9px', padding: '3px 6px' }}>✕</button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Art style selector (shared to generate/animate) */}
            {(imageGenSubTab === 'generate' || imageGenSubTab === 'animate') && (
              <div>
                <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>Art Style</label>
                <select value={imageGenArtStyle} onChange={e => setImageGenArtStyle(e.target.value)}
                  style={{ width: '100%', padding: '7px 10px', borderRadius: '8px', backgroundColor: '#111', color: '#fff', border: '1px solid #333', fontSize: '12px' }}>
                  {(artStyles.length > 0 ? artStyles : [{name:'anime',prefix:''},{name:'ghibli',prefix:''},{name:'manga',prefix:''},{name:'painterly',prefix:''},{name:'game_concept',prefix:''},{name:'realistic',prefix:''},{name:'custom',prefix:''}]).map(s => (
                    <option key={s.name} value={s.name}>{s.name}</option>
                  ))}
                </select>
              </div>
            )}

            {/* ──── Generate sub-tab controls ──── */}
            {imageGenSubTab === 'generate' && (<>
              <div>
                <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>Mode</label>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {(['normal', 'explicit'] as const).map(m => (
                    <button key={m} onClick={() => setImageGenMode(m)} style={{
                      flex: 1, padding: '6px', borderRadius: '8px', border: '1px solid',
                      borderColor: imageGenMode === m ? '#b388ff' : '#333',
                      backgroundColor: imageGenMode === m ? 'rgba(179,136,255,0.15)' : '#111',
                      color: imageGenMode === m ? '#b388ff' : '#888',
                      cursor: 'pointer', fontSize: '11px', textTransform: 'capitalize',
                    }}>{m}</button>
                  ))}
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Width</label>
                  <input type="number" value={imageGenWidth} onChange={e => setImageGenWidth(+e.target.value)} min={512} max={2048} step={64}
                    style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Height</label>
                  <input type="number" value={imageGenHeight} onChange={e => setImageGenHeight(+e.target.value)} min={512} max={2048} step={64}
                    style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Steps ({imageGenSteps})</label>
                  <input type="range" min={10} max={80} value={imageGenSteps} onChange={e => setImageGenSteps(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>CFG ({imageGenCfg.toFixed(1)})</label>
                  <input type="range" min={1} max={20} step={0.5} value={imageGenCfg} onChange={e => setImageGenCfg(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
                </div>
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Seed (-1 = random)</label>
                <input type="number" value={imageGenSeed} onChange={e => setImageGenSeed(+e.target.value)}
                  style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Custom Negative (optional)</label>
                <textarea value={imageGenNegative} onChange={e => setImageGenNegative(e.target.value)} placeholder="Leave empty for smart defaults..."
                  rows={2} style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '11px', resize: 'vertical', fontFamily: 'inherit' }} />
              </div>
              {/* Tag rating (Danbooru/E621 style) */}
              <div>
                <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>Tag Rating</label>
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                  {[
                    { val: '', label: 'None', color: '#666' },
                    { val: 'rating:general', label: '[G]', color: '#00cc88' },
                    { val: 'rating:sensitive', label: '[S]', color: '#ff9900' },
                    { val: 'rating:questionable', label: '[Q]', color: '#ff6644' },
                    { val: 'rating:explicit', label: '[E]', color: '#ff2222' },
                  ].map(r => (
                    <button key={r.val} onClick={() => setTagRating(r.val)} style={{
                      padding: '4px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: 'bold',
                      border: '1px solid', cursor: 'pointer',
                      borderColor: tagRating === r.val ? r.color : '#333',
                      backgroundColor: tagRating === r.val ? `${r.color}22` : '#111',
                      color: tagRating === r.val ? r.color : '#555',
                    }}>{r.label}</button>
                  ))}
                </div>
              </div>
            </>)}

            {/* ──── Animate sub-tab controls ──── */}
            {imageGenSubTab === 'animate' && (<>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Frames ({imageGenFrames})</label>
                  <input type="range" min={4} max={120} value={imageGenFrames} onChange={e => setImageGenFrames(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>FPS ({imageGenFps})</label>
                  <input type="range" min={2} max={30} value={imageGenFps} onChange={e => setImageGenFps(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
                </div>
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Frame Strength ({imageGenFrameStrength.toFixed(2)}) — lower = more consistent</label>
                <input type="range" min={0.1} max={0.7} step={0.05} value={imageGenFrameStrength} onChange={e => setImageGenFrameStrength(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Reference Blend ({animReferenceBlend.toFixed(2)}) — anchors to frame 0</label>
                <input type="range" min={0} max={0.6} step={0.05} value={animReferenceBlend} onChange={e => setAnimReferenceBlend(+e.target.value)} style={{ width: '100%', accentColor: '#00cc88' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Motion Intensity ({animMotionIntensity.toFixed(2)}) — subtle ↔ dramatic</label>
                <input type="range" min={0} max={1} step={0.05} value={animMotionIntensity} onChange={e => setAnimMotionIntensity(+e.target.value)} style={{ width: '100%', accentColor: '#ff9900' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Strength Curve</label>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {(['constant', 'ease_in', 'pulse'] as const).map(c => (
                    <button key={c} onClick={() => setAnimStrengthCurve(c)} style={{
                      flex: 1, padding: '5px', borderRadius: '6px', border: '1px solid',
                      borderColor: animStrengthCurve === c ? '#b388ff' : '#333',
                      backgroundColor: animStrengthCurve === c ? 'rgba(179,136,255,0.15)' : '#111',
                      color: animStrengthCurve === c ? '#b388ff' : '#888',
                      cursor: 'pointer', fontSize: '11px',
                    }}>{c.replace('_', ' ')}</button>
                  ))}
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Width</label>
                  <input type="number" value={imageGenWidth} onChange={e => setImageGenWidth(+e.target.value)} min={256} max={1024} step={64}
                    style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Height</label>
                  <input type="number" value={imageGenHeight} onChange={e => setImageGenHeight(+e.target.value)} min={256} max={1024} step={64}
                    style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
                </div>
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Steps ({imageGenSteps})</label>
                <input type="range" min={10} max={50} value={imageGenSteps} onChange={e => setImageGenSteps(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Output Format</label>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {(['gif', 'mp4', 'both'] as const).map(f => (
                    <button key={f} onClick={() => setImageGenOutputFormat(f)} style={{
                      flex: 1, padding: '5px', borderRadius: '6px', border: '1px solid',
                      borderColor: imageGenOutputFormat === f ? '#b388ff' : '#333',
                      backgroundColor: imageGenOutputFormat === f ? 'rgba(179,136,255,0.15)' : '#111',
                      color: imageGenOutputFormat === f ? '#b388ff' : '#888',
                      cursor: 'pointer', fontSize: '11px', textTransform: 'uppercase',
                    }}>{f}</button>
                  ))}
                </div>
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Storyboard (one action per line, optional)</label>
                <textarea value={storyboardDescs} onChange={e => setStoryboardDescs(e.target.value)}
                  placeholder={"e.g.:\ncharacter turns head left\ncharacter smiles\ncharacter waves hand"}
                  rows={4} style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '11px', resize: 'vertical', fontFamily: 'inherit' }} />
              </div>

              {/* Save State button */}
              <button onClick={async () => {
                try {
                  const res = await fetch('http://localhost:8000/generate/animated/save', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      prompt: imageGenPrompt, negative_prompt: imageGenNegative, art_style: imageGenArtStyle,
                      seed: imageGenSeed, num_frames: imageGenFrames, fps: imageGenFps,
                      frame_strength: imageGenFrameStrength, width: imageGenWidth, height: imageGenHeight,
                      steps: imageGenSteps, guidance_scale: imageGenCfg, output_format: imageGenOutputFormat,
                      storyboard_descriptions: storyboardDescs ? storyboardDescs.split('\n').filter(Boolean) : [],
                      reference_blend: animReferenceBlend, strength_curve: animStrengthCurve, motion_intensity: animMotionIntensity,
                    }),
                  });
                  if (res.ok) { fetchAnimJobs(); }
                } catch {}
              }} disabled={!imageGenPrompt.trim()} style={{
                width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #00cc8855',
                backgroundColor: 'rgba(0,204,136,0.08)', color: '#00cc88', cursor: imageGenPrompt.trim() ? 'pointer' : 'not-allowed',
                fontSize: '11px', fontWeight: 'bold',
              }}>💾 Save State</button>

              {/* Animation jobs */}
              {animJobs.length > 0 && (
                <div style={{ borderTop: '1px solid #222', paddingTop: '10px' }}>
                  <h4 style={{ margin: '0 0 6px', fontSize: '11px', color: '#888', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    Saved Jobs
                    <button onClick={fetchAnimJobs} style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: '10px' }}>⟳</button>
                  </h4>
                  {animJobs.map(j => (
                    <div key={j.job_id} style={{ padding: '6px', fontSize: '10px', color: '#aaa', borderRadius: '6px', backgroundColor: '#111', marginBottom: '4px', border: '1px solid #222' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                        <span>
                          <span style={{ color: j.status === 'complete' ? '#00cc88' : '#ff9900' }}>{j.status === 'complete' ? '✓' : '⏸'}</span>{' '}
                          {j.job_id.slice(0, 12)}... — {j.current_frame}/{j.total_frames}
                        </span>
                      </div>
                      <div style={{ color: '#666', fontSize: '9px', marginBottom: '4px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {j.prompt}
                      </div>
                      <div style={{ display: 'flex', gap: '4px' }}>
                        <button onClick={() => {
                          // Load saved state into form
                          setImageGenPrompt(j.prompt || '');
                          if (j.negative_prompt) setImageGenNegative(j.negative_prompt);
                          if (j.art_style) setImageGenArtStyle(j.art_style);
                          if (j.seed >= 0) setImageGenSeed(j.seed);
                          if (j.frame_strength) setImageGenFrameStrength(j.frame_strength);
                          if (j.fps) setImageGenFps(j.fps);
                          if (j.steps) setImageGenSteps(j.steps);
                          if (j.guidance_scale) setImageGenCfg(j.guidance_scale);
                          if (j.width) setImageGenWidth(j.width);
                          if (j.height) setImageGenHeight(j.height);
                          if (j.total_frames) setImageGenFrames(j.total_frames);
                          if (j.output_format) setImageGenOutputFormat(j.output_format);
                          if (j.storyboard?.length > 0) setStoryboardDescs(j.storyboard.join('\n'));
                          if (j.reference_blend != null) setAnimReferenceBlend(j.reference_blend);
                          if (j.strength_curve) setAnimStrengthCurve(j.strength_curve);
                          if (j.motion_intensity != null) setAnimMotionIntensity(j.motion_intensity);
                        }} style={{ flex: 1, padding: '3px', borderRadius: '4px', border: '1px solid #333', backgroundColor: 'transparent', color: '#b388ff', cursor: 'pointer', fontSize: '9px' }}
                          title="Load this job's prompt and settings into the form">📋 Load</button>
                        {j.status !== 'complete' && (
                          <button onClick={() => {
                            // Load state and auto-resume
                            setImageGenPrompt(j.prompt || '');
                            if (j.negative_prompt) setImageGenNegative(j.negative_prompt);
                            if (j.art_style) setImageGenArtStyle(j.art_style);
                            if (j.seed >= 0) setImageGenSeed(j.seed);
                            if (j.frame_strength) setImageGenFrameStrength(j.frame_strength);
                            if (j.fps) setImageGenFps(j.fps);
                            if (j.steps) setImageGenSteps(j.steps);
                            if (j.guidance_scale) setImageGenCfg(j.guidance_scale);
                            if (j.width) setImageGenWidth(j.width);
                            if (j.height) setImageGenHeight(j.height);
                            if (j.total_frames) setImageGenFrames(j.total_frames);
                            if (j.output_format) setImageGenOutputFormat(j.output_format);
                            if (j.storyboard?.length > 0) setStoryboardDescs(j.storyboard.join('\n'));
                            if (j.reference_blend != null) setAnimReferenceBlend(j.reference_blend);
                            if (j.strength_curve) setAnimStrengthCurve(j.strength_curve);
                            if (j.motion_intensity != null) setAnimMotionIntensity(j.motion_intensity);
                            // Trigger resume generation
                            setTimeout(() => handleImageGenerate(true), 100);
                          }} style={{ flex: 1, padding: '3px', borderRadius: '4px', border: '1px solid #00cc88', backgroundColor: 'rgba(0,204,136,0.1)', color: '#00cc88', cursor: 'pointer', fontSize: '9px' }}
                            title="Resume generating from last frame">▶ Resume</button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>)}

            {/* ──── Storyboard sub-tab ──── */}
            {imageGenSubTab === 'storyboard' && (<>
              <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.5' }}>
                Generate a quick low-res sketch grid to preview your animation before committing to a full render.
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Frames</label>
                <input type="number" value={imageGenFrames} onChange={e => setImageGenFrames(+e.target.value)} min={2} max={24}
                  style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Frame Descriptions (one per line)</label>
                <textarea value={storyboardDescs} onChange={e => setStoryboardDescs(e.target.value)}
                  placeholder={"Character standing still\nCharacter raises hand\nCharacter waves"}
                  rows={6} style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '11px', resize: 'vertical', fontFamily: 'inherit' }} />
              </div>
              <button onClick={handleStoryboardPreview} disabled={!imageGenPrompt.trim() || imageGenLoading}
                style={{ width: '100%', padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: imageGenLoading ? '#333' : '#7c4dff', color: '#fff', cursor: imageGenLoading ? 'wait' : 'pointer', fontSize: '13px', fontWeight: 'bold' }}>
                {imageGenLoading ? '⏳ Sketching...' : '📋 Generate Storyboard'}
              </button>
            </>)}

            {/* ──── Upscale sub-tab ──── */}
            {imageGenSubTab === 'upscale' && (<>
              <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.5' }}>
                Tile-based AI upscaler. Works within 8 GB VRAM by processing the image in overlapping tiles, then stitching them back seamlessly.
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Image Path</label>
                <div style={{ display: 'flex', gap: '6px' }}>
                  <input type="text" value={upscaleImagePath} onChange={e => setUpscaleImagePath(e.target.value)}
                    placeholder="C:\path\to\image.png"
                    style={{ flex: 1, padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
                  <input id="upscale-file-picker" type="file" accept="image/png,image/jpeg,image/webp" style={{ display: 'none' }}
                    onChange={(e) => {
                      const f = (e.target as HTMLInputElement).files?.[0];
                      if (!f) return;
                      const anyF = f as any;
                      if (anyF.path) {
                        setUpscaleImagePath(anyF.path);
                      } else {
                        setUpscaleImagePath(f.name);
                      }
                      (document.getElementById('upscale-file-picker') as HTMLInputElement).value = '';
                    }}
                  />
                  <button onClick={() => (document.getElementById('upscale-file-picker') as HTMLInputElement)?.click()}
                    style={{ padding: '6px 12px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#b388ff', cursor: 'pointer', fontSize: '12px', whiteSpace: 'nowrap' }}
                    title="Browse for image file">📁 Browse</button>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Scale ({upscaleScale.toFixed(1)}x)</label>
                  <input type="range" min={1.5} max={4} step={0.5} value={upscaleScale} onChange={e => setUpscaleScale(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Tile Size</label>
                  <select value={upscaleTileSize} onChange={e => setUpscaleTileSize(+e.target.value)}
                    style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }}>
                    <option value={512}>512 (low VRAM)</option>
                    <option value={768}>768 (recommended)</option>
                    <option value={1024}>1024 (fast)</option>
                  </select>
                </div>
              </div>
              <button onClick={handleUpscale} disabled={!upscaleImagePath.trim() || imageGenLoading}
                style={{ width: '100%', padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: imageGenLoading ? '#333' : '#7c4dff', color: '#fff', cursor: imageGenLoading ? 'wait' : 'pointer', fontSize: '13px', fontWeight: 'bold' }}>
                {imageGenLoading ? '⏳ Upscaling...' : '🔍 Upscale Image'}
              </button>
            </>)}

            {/* ──── Train sub-tab ──── */}
            {imageGenSubTab === 'train' && (<>
              <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.5' }}>
                Train a LoRA from reference images to learn a specific art style or character. Trained LoRAs appear in the LoRA Adapters list.
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>LoRA Name</label>
                <input type="text" value={trainingName} onChange={e => setTrainingName(e.target.value)} placeholder="my_artstyle"
                  style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Image Folder Path</label>
                <input type="text" value={trainingImageDir} onChange={e => setTrainingImageDir(e.target.value)}
                  placeholder="C:\path\to\training_images"
                  style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Training Type</label>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {(['style', 'character'] as const).map(t => (
                    <button key={t} onClick={() => setTrainingType(t)} style={{
                      flex: 1, padding: '5px', borderRadius: '6px', border: '1px solid',
                      borderColor: trainingType === t ? '#b388ff' : '#333',
                      backgroundColor: trainingType === t ? 'rgba(179,136,255,0.15)' : '#111',
                      color: trainingType === t ? '#b388ff' : '#888',
                      cursor: 'pointer', fontSize: '11px', textTransform: 'capitalize',
                    }}>{t}</button>
                  ))}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button onClick={handleCritiqueDataset} disabled={!trainingImageDir.trim()}
                  style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid #ff9900', backgroundColor: 'transparent', color: '#ff9900', cursor: trainingImageDir.trim() ? 'pointer' : 'not-allowed', fontSize: '12px' }}>
                  🔎 Critique Dataset
                </button>
                <button onClick={handleStartTraining} disabled={!trainingName.trim() || !trainingImageDir.trim()}
                  style={{ flex: 1, padding: '8px', borderRadius: '8px', border: 'none', backgroundColor: (!trainingName.trim() || !trainingImageDir.trim()) ? '#333' : '#7c4dff', color: '#fff', cursor: (!trainingName.trim() || !trainingImageDir.trim()) ? 'not-allowed' : 'pointer', fontSize: '12px', fontWeight: 'bold' }}>
                  🚀 Train
                </button>
              </div>

              {/* Dataset critique results */}
              {trainingCritique && trainingCritique.status === 'ok' && (
                <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: '#0a0a0a', border: '1px solid #222', fontSize: '11px' }}>
                  <div style={{ color: trainingCritique.quality === 'good' ? '#00cc88' : trainingCritique.quality === 'needs_work' ? '#ff9900' : '#ff4444', fontWeight: 'bold', marginBottom: '6px' }}>
                    {trainingCritique.quality === 'good' ? '✓ Dataset looks good' : trainingCritique.quality === 'needs_work' ? '⚠ Needs improvement' : '✗ Insufficient'}
                  </div>
                  <div style={{ color: '#888' }}>{trainingCritique.image_count} images (min: {trainingCritique.minimum_recommended})</div>
                  {trainingCritique.issues?.map((issue: string, i: number) => (
                    <div key={i} style={{ color: '#ff6666', marginTop: '3px' }}>• {issue}</div>
                  ))}
                  {trainingCritique.suggestions?.map((s: string, i: number) => (
                    <div key={i} style={{ color: '#66aaff', marginTop: '3px' }}>💡 {s}</div>
                  ))}
                </div>
              )}

              {/* Training status */}
              {trainingStatus && trainingStatus.status === 'training' && (
                <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: '#0a0a0a', border: '1px solid #b388ff33' }}>
                  <div style={{ fontSize: '12px', color: '#b388ff', marginBottom: '6px' }}>Training: {trainingStatus.job_name}</div>
                  <div style={{ height: '6px', borderRadius: '3px', backgroundColor: '#222', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${trainingStatus.progress}%`, backgroundColor: '#7c4dff', transition: 'width 0.5s' }} />
                  </div>
                  <div style={{ fontSize: '10px', color: '#888', marginTop: '4px' }}>
                    Epoch {trainingStatus.current_epoch}/{trainingStatus.total_epochs} — {trainingStatus.progress}%
                  </div>
                  <button onClick={fetchTrainingStatus} style={{ marginTop: '4px', padding: '3px 8px', borderRadius: '4px', border: '1px solid #555', backgroundColor: 'transparent', color: '#888', cursor: 'pointer', fontSize: '10px' }}>⟳ Refresh</button>
                </div>
              )}
            </>)}

            {/* ──── Video sub-tab (WAN) ──── */}
            {imageGenSubTab === 'video' && (<>
              <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.5' }}>
                AI video generation using WAN 2.1. Supports text-to-video and image-to-video with dual-LoRA actions (high + low noise).
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Reference Image (optional — enables I2V mode)</label>
                <input type="text" value={videoRefImage} onChange={e => setVideoRefImage(e.target.value)}
                  placeholder="C:\path\to\reference.png (leave empty for text-to-video)"
                  style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Frames ({videoFrames})</label>
                  <input type="range" min={17} max={200} step={4} value={videoFrames} onChange={e => setVideoFrames(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>FPS ({videoFps})</label>
                  <input type="range" min={8} max={30} value={videoFps} onChange={e => setVideoFps(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Width</label>
                  <input type="number" value={imageGenWidth} onChange={e => setImageGenWidth(+e.target.value)} min={128} max={1280} step={16}
                    style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: '#888' }}>Height</label>
                  <input type="number" value={imageGenHeight} onChange={e => setImageGenHeight(+e.target.value)} min={128} max={1280} step={16}
                    style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px' }} />
                </div>
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Steps ({videoSteps})</label>
                <input type="range" min={10} max={50} value={videoSteps} onChange={e => setVideoSteps(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>CFG Scale ({videoCfg.toFixed(1)})</label>
                <input type="range" min={1} max={15} step={0.5} value={videoCfg} onChange={e => setVideoCfg(+e.target.value)} style={{ width: '100%', accentColor: '#b388ff' }} />
              </div>
              <div>
                <label style={{ fontSize: '11px', color: '#888' }}>Action LoRA (auto-pairs high+low noise)</label>
                <select value={videoActionLora} onChange={e => setVideoActionLora(e.target.value)}
                  style={{ width: '100%', padding: '7px', borderRadius: '6px', backgroundColor: '#111', color: '#fff', border: '1px solid #333', fontSize: '12px' }}>
                  <option value="">None</option>
                  {actionLoras.map((l: any) => (
                    <option key={l.name} value={l.name}>
                      {l.noise_type === 'high_noise' ? '⬆ ' : l.noise_type === 'low_noise' ? '⬇ ' : ''}{l.name} ({l.size_mb}MB){l.has_pair ? ' [paired]' : ''}
                    </option>
                  ))}
                </select>
                <button onClick={async () => {
                  try {
                    const r = await fetch('http://localhost:8000/generate/video/action-loras');
                    if (r.ok) { const d = await r.json(); setActionLoras(d.loras || []); }
                  } catch {}
                }} style={{ marginTop: '4px', padding: '3px 8px', borderRadius: '4px', border: '1px solid #555', backgroundColor: 'transparent', color: '#888', cursor: 'pointer', fontSize: '10px' }}>⟳ Refresh LoRAs</button>
              </div>
              <button onClick={async () => {
                if (!imageGenPrompt.trim() || imageGenLoading) return;
                setImageGenLoading(true);
                setImageGenResult(null);
                setIsVideoResult(false);
                setMode('querying');
                const controller = new AbortController();
                globalAbortRef.current = controller;
                try {
                  const res = await fetch('http://localhost:8000/generate/video', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    signal: controller.signal,
                    body: JSON.stringify({
                      prompt: (tagRating ? tagRating + ', ' : '') + imageGenPrompt.trim(),
                      image_path: videoRefImage ? videoRefImage.replace(/^["']|["']$/g, '').trim() : undefined,
                      width: imageGenWidth, height: imageGenHeight,
                      num_frames: videoFrames, fps: videoFps,
                      steps: videoSteps, guidance_scale: videoCfg,
                      seed: imageGenSeed,
                      action_lora: videoActionLora || undefined,
                      negative_prompt: imageGenNegative.trim() || undefined,
                    }),
                  });
                  if (res.ok) {
                    const ct = res.headers.get('content-type') ?? '';
                    if (ct.includes('video')) {
                      const blob = await res.blob();
                      setIsVideoResult(true);
                      setImageGenResult(URL.createObjectURL(blob));
                    } else if (ct.includes('json')) {
                      const data = await res.json();
                      setImageGenResult(`Error: ${data.error || 'Unexpected response'}`);
                    }
                  } else {
                    const err = await res.json().catch(() => ({}));
                    setImageGenResult(`Error: ${err.error || 'Video generation failed'}`);
                  }
                } catch (e: any) {
                  if (e?.name !== 'AbortError') setImageGenResult(`Error: ${e}`);
                } finally {
                  globalAbortRef.current = null;
                  setImageGenLoading(false);
                  setMode('idle');
                }
              }} disabled={!imageGenPrompt.trim() || imageGenLoading} style={{
                width: '100%', padding: '10px', borderRadius: '8px', border: 'none',
                backgroundColor: imageGenLoading ? '#333' : '#7c4dff', color: '#fff',
                cursor: imageGenLoading ? 'wait' : 'pointer', fontSize: '13px', fontWeight: 'bold',
              }}>
                {imageGenLoading ? '⏳ Generating Video...' : '🎥 Generate Video'}
              </button>
              {/* Pause / Resume / Cancel controls during video gen */}
              {imageGenLoading && imageGenSubTab === 'video' && (
                <div style={{ display: 'flex', gap: '6px', marginTop: '4px' }}>
                  <button onClick={async () => {
                    try {
                      const isPaused = genProgress.message?.includes('PAUSED');
                      const endpoint = isPaused ? 'resume' : 'pause';
                      await fetch(`http://localhost:8000/generate/video/${endpoint}`, { method: 'POST' });
                    } catch {}
                  }} style={{
                    flex: 1, padding: '6px', borderRadius: '6px', border: '1px solid #555',
                    backgroundColor: 'transparent', cursor: 'pointer', fontSize: '11px', fontWeight: 'bold',
                    color: genProgress.message?.includes('PAUSED') ? '#00cc88' : '#ff9900',
                  }}>
                    {genProgress.message?.includes('PAUSED') ? '▶ Resume' : '⏸ Pause'}
                  </button>
                  <button onClick={handleCancelGeneration} style={{
                    flex: 1, padding: '6px', borderRadius: '6px', border: '1px solid #ff444466',
                    backgroundColor: 'transparent', color: '#ff4444', cursor: 'pointer', fontSize: '11px', fontWeight: 'bold',
                  }}>
                    ✕ Cancel
                  </button>
                </div>
              )}
              {/* Resumable checkpoint jobs */}
              {interruptedJobs.length > 0 && !imageGenLoading && (
                <div style={{ marginTop: '6px', padding: '8px', borderRadius: '8px', border: '1px solid #ff990033', backgroundColor: '#1a1200' }}>
                  <div style={{ fontSize: '11px', color: '#ff9900', marginBottom: '4px', fontWeight: 'bold' }}>📦 Resumable Jobs ({interruptedJobs.length})</div>
                  {interruptedJobs.slice(0, 5).map((job: any, i: number) => {
                    const savedSteps = job.checkpoint_count || job.latest_checkpoint || 0;
                    const totalSteps = job.steps || job.total_steps || '?';
                    const allDone = savedSteps >= totalSteps;
                    return (<div key={job.job_id || i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '10px', color: '#999', padding: '4px 0', borderTop: i > 0 ? '1px solid #222' : 'none' }}>
                      <div style={{ flex: 1, overflow: 'hidden', minWidth: 0 }}>
                        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {job.prompt?.slice(0, 50) || job.job_id}
                        </div>
                        <div style={{ fontSize: '9px', color: allDone ? '#00cc88' : '#888' }}>
                          {allDone ? `✓ ${savedSteps}/${totalSteps} steps — ready to decode (~30s)` : `${savedSteps}/${totalSteps} steps — will re-run from scratch`}
                          {' · '}{job.status}
                        </div>
                      </div>
                      <button onClick={async () => {
                        if (imageGenLoading) return;
                        setImageGenLoading(true);
                        setImageGenResult(null);
                        setIsVideoResult(false);
                        setMode('querying');
                        const controller = new AbortController();
                        globalAbortRef.current = controller;
                        try {
                          const res = await fetch('http://localhost:8000/generate/video/resume-job', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            signal: controller.signal,
                            body: JSON.stringify({ job_id: job.job_id }),
                          });
                          if (res.ok) {
                            const ct = res.headers.get('content-type') ?? '';
                            if (ct.includes('video')) {
                              const blob = await res.blob();
                              setIsVideoResult(true);
                              setImageGenResult(URL.createObjectURL(blob));
                            } else if (ct.includes('json')) {
                              const data = await res.json();
                              if (data.status === 'ok' && data.path) {
                                setImageGenResult(data.path);
                              } else {
                                setImageGenResult(`Error: ${data.error || 'Unexpected response'}`);
                              }
                            }
                          } else {
                            const err = await res.json().catch(() => ({}));
                            setImageGenResult(`Error: ${err.error || 'Resume failed'}`);
                          }
                        } catch (e: any) {
                          if (e?.name !== 'AbortError') setImageGenResult(`Error: ${e}`);
                        } finally {
                          globalAbortRef.current = null;
                          setImageGenLoading(false);
                          setMode('idle');
                        }
                      }} style={{ marginLeft: '6px', padding: '2px 10px', borderRadius: '4px', border: '1px solid #00cc8866', backgroundColor: 'transparent', color: '#00cc88', cursor: 'pointer', fontSize: '10px', fontWeight: 'bold' }}>
                        ▶ Resume
                      </button>
                    </div>
                  );
                  })}
                  {interruptedJobs.length > 3 && <div style={{ fontSize: '9px', color: '#666', marginTop: '2px' }}>+{interruptedJobs.length - 3} more</div>}
                </div>
              )}
              <div style={{ fontSize: '10px', color: '#555', lineHeight: 1.4 }}>
                WAN 2.1 T2V (1.3B) — Loaded directly on GPU (~2.6GB model + working memory).
                Falls back to CPU offload if VRAM is insufficient. Frames saved as PNGs if mp4 export fails.
              </div>
            </>)}

            {/* Model management (collapsible) */}
            <div style={{ borderTop: '1px solid #222', paddingTop: '12px', marginTop: '4px' }}>
              <button onClick={() => setModelsExpanded(!modelsExpanded)} style={{
                width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '6px 8px', borderRadius: '6px', border: '1px solid #333',
                backgroundColor: '#111', color: '#888', cursor: 'pointer', fontSize: '12px',
              }}>
                <span>🧠 Models ({imageModels.length})</span>
                <span style={{ fontSize: '10px' }}>{modelsExpanded ? '▲' : '▼'}</span>
              </button>
              {modelsExpanded && (
                <div style={{ marginTop: '6px' }}>
                  {imageModels.map(m => (
                    <div key={m.path} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0', fontSize: '11px', color: '#ccc' }}>
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {m.type === 'lora' ? '🔗 ' : '🧠 '}{m.name}
                        <span style={{ color: '#666', marginLeft: '4px' }}>({m.size_mb}MB)</span>
                      </span>
                      <button onClick={() => deleteImageModel(m.path)} title="Delete model"
                        style={{ background: 'none', border: 'none', color: '#ff4444', cursor: 'pointer', fontSize: '12px', padding: '2px 6px' }}>🗑</button>
                    </div>
                  ))}
                  {imageModels.length === 0 && <div style={{ fontSize: '11px', color: '#555' }}>No models in models/</div>}
                  <div style={{ fontSize: '10px', color: '#555', marginTop: '6px' }}>Drop .safetensors into models/. LoRAs: models/loras/</div>
                </div>
              )}
            </div>
          </div>

          {/* Center panel: Prompt + Result */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', minWidth: 0 }}>
            {/* Prompt input */}
            <div style={{
              backgroundColor: 'rgba(10,10,20,0.95)', borderRadius: '14px',
              border: '1px solid rgba(179,136,255,0.2)', backdropFilter: 'blur(16px)',
              padding: '14px', display: 'flex', gap: '10px', alignItems: 'flex-start',
            }}>
              <textarea
                value={imageGenPrompt}
                onChange={e => { setImageGenPrompt(e.target.value); updateTokenCount(e.target.value); }}
                onBlur={() => fetchVocabExpansion(imageGenPrompt)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (imageGenSubTab === 'storyboard') handleStoryboardPreview(); else handleImageGenerate(imageGenSubTab === 'animate'); } }}
                placeholder="Describe what to generate... (supports long prompts, no 77-token limit, use (word:1.5) for weighting)"
                rows={3}
                style={{
                  flex: 1, padding: '10px 14px', borderRadius: '10px', border: '1px solid #333',
                  backgroundColor: '#0a0a0a', color: '#eee', fontSize: '14px', outline: 'none',
                  resize: 'vertical', fontFamily: 'inherit', lineHeight: '1.5',
                }}
              />
              <button
                disabled={!imageGenPrompt.trim() || imageGenLoading}
                onClick={() => {
                  if (imageGenSubTab === 'animate') { handleImageGenerate(true); }
                  else if (imageGenSubTab === 'storyboard') { handleStoryboardPreview(); }
                  else if (imageGenSubTab === 'video') { /* video has its own button */ }
                  else { handleImageGenerate(false); }
                }}
                style={{
                  padding: '12px 24px', borderRadius: '10px', border: 'none',
                  backgroundColor: imageGenLoading ? '#333' : '#7c4dff', color: '#fff',
                  cursor: imageGenLoading ? 'wait' : 'pointer', fontSize: '14px', fontWeight: 'bold',
                  whiteSpace: 'nowrap', transition: 'background 0.2s',
                }}
              >
                {imageGenLoading ? '⏳ Working...'
                  : imageGenSubTab === 'animate' ? '🎬 Animate'
                  : imageGenSubTab === 'storyboard' ? '📋 Sketch'
                  : imageGenSubTab === 'video' ? '🎥 Video'
                  : '🎨 Generate'}
              </button>
              {imageGenSubTab === 'generate' && (
                <button
                  disabled={!imageGenPrompt.trim() || previewLoading || imageGenLoading}
                  onClick={handleImagePreview}
                  style={{
                    padding: '12px 18px', borderRadius: '10px', border: '1px solid #ff990066',
                    backgroundColor: previewLoading ? '#333' : 'rgba(255,153,0,0.1)', color: '#ff9900',
                    cursor: (previewLoading || imageGenLoading) ? 'wait' : 'pointer', fontSize: '13px', fontWeight: 'bold',
                    whiteSpace: 'nowrap', transition: 'background 0.2s',
                  }}
                >
                  {previewLoading ? '⏳ Preview...' : '👁 Preview'}
                </button>
              )}
              {imageGenLoading && (
                <button onClick={handleCancelGeneration}
                  style={{
                    padding: '12px 16px', borderRadius: '10px', border: '2px solid #ff4444',
                    backgroundColor: 'rgba(255,68,68,0.1)', color: '#ff4444',
                    cursor: 'pointer', fontSize: '14px', fontWeight: 'bold',
                    whiteSpace: 'nowrap',
                  }}>
                  ⬛ Stop
                </button>
              )}
              {!imageGenLoading && (
                <button onClick={handleFlushVram} disabled={vramFlushing}
                  style={{
                    padding: '12px 16px', borderRadius: '10px', border: '2px solid #44aaff',
                    backgroundColor: vramFlushing ? '#333' : 'rgba(68,170,255,0.1)', color: '#44aaff',
                    cursor: vramFlushing ? 'wait' : 'pointer', fontSize: '13px', fontWeight: 'bold',
                    whiteSpace: 'nowrap', transition: 'background 0.2s',
                  }}>
                  {vramFlushing ? '⏳ Freeing...' : '🧹 Free VRAM'}
                </button>
              )}
            </div>

            {/* Token counter bar */}
            {imageGenPrompt.trim() && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '6px 14px', fontSize: '11px', fontFamily: 'monospace',
              }}>
                <div style={{ flex: 1, height: '4px', borderRadius: '2px', backgroundColor: '#1a1a2e', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: '2px', transition: 'width 0.3s, background 0.3s',
                    width: `${Math.min(100, (promptTokenCount / 225) * 100)}%`,
                    background: promptTokenCount <= 75 ? '#00cc88' : promptTokenCount <= 150 ? '#ff9900' : '#ff4444',
                  }} />
                </div>
                <span style={{
                  color: promptTokenCount <= 75 ? '#00cc88' : promptTokenCount <= 150 ? '#ff9900' : '#ff4444',
                  whiteSpace: 'nowrap', minWidth: '80px',
                }}>
                  {promptTokenCount} / 75 tokens
                </span>
                {promptTokenCount > 75 && (
                  <span style={{ color: '#00cc88', fontSize: '10px', whiteSpace: 'nowrap' }}>
                    ✓ Auto-chunked ({Math.ceil(promptTokenCount / 75)} segments)
                  </span>
                )}
                {promptTokenMethod === 'clip_tokenizer' && (
                  <span style={{ color: '#555', fontSize: '9px' }}>CLIP</span>
                )}
              </div>
            )}

            {/* Vocab expansion suggestions */}
            {Object.keys(vocabSuggestions).length > 0 && (
              <div style={{
                backgroundColor: 'rgba(10,10,20,0.95)', borderRadius: '10px',
                border: '1px solid rgba(179,136,255,0.15)', padding: '10px',
                fontSize: '11px',
              }}>
                <div style={{ color: '#888', marginBottom: '6px', fontSize: '10px' }}>💡 Synonym suggestions (click to insert):</div>
                {Object.entries(vocabSuggestions).map(([word, synonyms]) => (
                  <div key={word} style={{ marginBottom: '4px' }}>
                    <span style={{ color: '#b388ff' }}>{word}</span>
                    <span style={{ color: '#555' }}> → </span>
                    {(synonyms as string[]).map((s, i) => (
                      <span key={i}>
                        <span
                          onClick={() => setImageGenPrompt(prev => prev + ', ' + s)}
                          style={{ color: '#00cc88', cursor: 'pointer', textDecoration: 'underline' }}
                        >{s}</span>
                        {i < (synonyms as string[]).length - 1 && <span style={{ color: '#333' }}> | </span>}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            )}

            {/* GPU monitor + Generation progress */}
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {/* GPU VRAM bar */}
              {gpuInfo && (gpuInfo.vram_total_mb > 0 ? (
                <div style={{
                  flex: 1, minWidth: '180px', padding: '6px 10px', borderRadius: '8px',
                  backgroundColor: '#0a0a14', border: `1px solid ${gpuInfo.vram_percent > 90 ? '#ff444466' : gpuInfo.vram_percent > 75 ? '#ff990033' : '#222'}`,
                  fontSize: '10px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                    <span style={{ color: '#888' }}>🖥 {gpuInfo.device_name || 'GPU'}</span>
                    <span style={{ color: gpuInfo.vram_percent > 90 ? '#ff4444' : gpuInfo.vram_percent > 75 ? '#ff9900' : '#00cc88' }}>
                      {gpuInfo.vram_used_mb.toFixed(0)} / {gpuInfo.vram_total_mb.toFixed(0)} MB ({gpuInfo.vram_percent.toFixed(1)}%)
                    </span>
                  </div>
                  <div style={{ height: '4px', borderRadius: '2px', backgroundColor: '#1a1a2e', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', borderRadius: '2px', transition: 'width 0.5s, background 0.3s',
                      width: `${gpuInfo.vram_percent}%`,
                      background: gpuInfo.vram_percent > 90 ? '#ff4444' : gpuInfo.vram_percent > 75 ? '#ff9900' : '#00cc88',
                    }} />
                  </div>
                </div>
              ) : (
                <div style={{
                  flex: 1, minWidth: '180px', padding: '6px 10px', borderRadius: '8px',
                  backgroundColor: '#0a0a14', border: '1px solid #222', fontSize: '10px',
                }}>
                  <span style={{ color: '#555' }}>🖥 {gpuInfo.device_name || 'GPU'} — {gpuInfo.error || (gpuInfo.available === false ? 'CUDA not available' : 'No VRAM data')}</span>
                </div>
              ))}
              {/* Generation progress bar */}
              {genProgress.active && (
                <div style={{
                  flex: 1, minWidth: '180px', padding: '6px 10px', borderRadius: '8px',
                  backgroundColor: '#0a0a14', border: '1px solid #b388ff33',
                  fontSize: '10px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                    <span style={{ color: '#b388ff' }}>
                      {genProgress.type === 'animation'
                        ? `🎬 Frame ${genProgress.current_frame + 1}/${genProgress.total_frames}`
                        : genProgress.type === 'video' ? '🎥 Video'
                        : '🎨 Generating'}
                    </span>
                    <span style={{ color: '#aaa' }}>
                      Step {genProgress.current_step}/{genProgress.total_steps}
                      {genProgress.total_steps > 0 && ` (${Math.round(genProgress.current_step / genProgress.total_steps * 100)}%)`}
                    </span>
                  </div>
                  <div style={{ height: '4px', borderRadius: '2px', backgroundColor: '#1a1a2e', overflow: 'hidden' }}>
                    {genProgress.type === 'animation' ? (
                      /* Two-layer bar: frame progress (outer) + step progress (inner shimmer) */
                      <div style={{ height: '100%', display: 'flex' }}>
                        <div style={{
                          height: '100%', borderRadius: '2px', transition: 'width 0.3s',
                          width: `${(genProgress.current_frame / Math.max(1, genProgress.total_frames)) * 100}%`,
                          background: '#7c4dff',
                        }} />
                        <div style={{
                          height: '100%', transition: 'width 0.3s',
                          width: `${(genProgress.current_step / Math.max(1, genProgress.total_steps)) * (100 / Math.max(1, genProgress.total_frames))}%`,
                          background: '#b388ff88',
                        }} />
                      </div>
                    ) : (
                      <div style={{
                        height: '100%', borderRadius: '2px', transition: 'width 0.3s',
                        width: `${genProgress.total_steps > 0 ? (genProgress.current_step / genProgress.total_steps * 100) : 0}%`,
                        background: '#7c4dff',
                      }} />
                    )}
                  </div>
                  {genProgress.message && (
                    <div style={{ color: '#666', marginTop: '2px', fontSize: '9px' }}>{genProgress.message}</div>
                  )}
                </div>
              )}
            </div>

            {/* Result area */}
            <div style={{
              flex: 1, backgroundColor: 'rgba(10,10,20,0.95)', borderRadius: '14px',
              border: '1px solid rgba(179,136,255,0.2)', backdropFilter: 'blur(16px)',
              overflow: 'hidden', display: 'flex', flexDirection: 'column',
            }}>
              {(imageGenLoading || previewLoading) && (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '12px', padding: '20px' }}>
                  <div style={{ fontSize: '32px', animation: 'spin 2s linear infinite' }}>{previewLoading ? '👁' : genProgress.type === 'video' ? '🎥' : '🎨'}</div>
                  <div style={{ color: previewLoading ? '#ff9900' : '#b388ff', fontSize: '14px' }}>
                    {previewLoading ? 'Generating quick preview...'
                      : genProgress.type === 'video' ? 'Generating video...'
                      : imageGenAnimated ? `Generating ${imageGenFrames} frames...` : 'Generating full image...'}
                  </div>
                  {/* Inline progress for video generation */}
                  {genProgress.active && genProgress.type === 'video' && (
                    <div style={{ width: '100%', maxWidth: '400px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                        <span style={{ color: '#b388ff' }}>Step {genProgress.current_step}/{genProgress.total_steps}</span>
                        <span style={{ color: '#aaa' }}>
                          {genProgress.total_steps > 0 ? `${Math.round(genProgress.current_step / genProgress.total_steps * 100)}%` : '0%'}
                        </span>
                      </div>
                      <div style={{ height: '8px', borderRadius: '4px', backgroundColor: '#1a1a2e', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%', borderRadius: '4px', transition: 'width 0.5s ease',
                          width: `${genProgress.total_steps > 0 ? (genProgress.current_step / genProgress.total_steps * 100) : 0}%`,
                          background: 'linear-gradient(90deg, #7c4dff, #b388ff)',
                        }} />
                      </div>
                      {genProgress.message && (
                        <div style={{ color: '#888', marginTop: '6px', fontSize: '11px', textAlign: 'center' }}>{genProgress.message}</div>
                      )}
                    </div>
                  )}
                  <div style={{ color: '#555', fontSize: '11px' }}>
                    {previewLoading ? 'Fast noisy preview — a few seconds'
                      : genProgress.type === 'video' ? 'Model offload + attention slicing — GPU-accelerated on 8GB VRAM'
                      : 'This may take 10-30 seconds depending on settings'}
                  </div>
                </div>
              )}

              {/* Preview result with Full Render button */}
              {!imageGenLoading && !previewLoading && previewResult && !imageGenResult && (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '16px', gap: '12px' }}>
                  <div style={{ fontSize: '11px', color: '#ff9900', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold' }}>
                    Preview (low-res, {previewSeed !== null ? `seed: ${previewSeed}` : 'noisy'})
                  </div>
                  <img
                    src={previewResult}
                    alt="Preview"
                    onClick={() => { setLightboxUrl(previewResult); setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 }); }}
                    style={{ maxWidth: '100%', maxHeight: 'calc(100% - 80px)', borderRadius: '8px', cursor: 'zoom-in', objectFit: 'contain', imageRendering: 'auto', border: '2px solid rgba(255,153,0,0.3)' }}
                  />
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                      onClick={handleFullRenderFromPreview}
                      style={{
                        padding: '10px 24px', borderRadius: '10px', border: 'none',
                        backgroundColor: '#7c4dff', color: '#fff', fontWeight: 'bold',
                        cursor: 'pointer', fontSize: '14px', transition: 'background 0.2s',
                      }}
                    >
                      🎨 Full Render (seed {previewSeed})
                    </button>
                    <button
                      onClick={() => { setPreviewResult(null); setPreviewSeed(null); }}
                      style={{
                        padding: '10px 16px', borderRadius: '10px', border: '1px solid #555',
                        backgroundColor: 'transparent', color: '#888',
                        cursor: 'pointer', fontSize: '13px',
                      }}
                    >
                      Discard
                    </button>
                    <button
                      onClick={handleImagePreview}
                      style={{
                        padding: '10px 16px', borderRadius: '10px', border: '1px solid #ff990044',
                        backgroundColor: 'rgba(255,153,0,0.08)', color: '#ff9900',
                        cursor: 'pointer', fontSize: '13px',
                      }}
                    >
                      ↻ Re-roll
                    </button>
                  </div>
                </div>
              )}

              {!imageGenLoading && !previewLoading && imageGenResult && (
                imageGenResult.startsWith('Error') ? (
                  <div style={{ padding: '20px', color: '#ff6666', fontSize: '14px' }}>{imageGenResult}</div>
                ) : imageGenResult.startsWith('blob:') && isVideoResult ? (
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px', position: 'relative' }}>
                    <video
                      src={imageGenResult}
                      controls
                      autoPlay
                      loop
                      style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: '8px', objectFit: 'contain' }}
                    />
                  </div>
                ) : imageGenResult.startsWith('blob:') ? (
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px', position: 'relative' }}>
                    <img
                      src={imageGenResult}
                      alt="Generated"
                      onClick={() => { setLightboxUrl(imageGenResult); setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 }); }}
                      style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: '8px', cursor: 'zoom-in', objectFit: 'contain' }}
                    />
                  </div>
                ) : (
                  <div style={{ padding: '20px', color: '#ccc', fontSize: '13px', whiteSpace: 'pre-wrap' }}>{imageGenResult}</div>
                )
              )}

              {!imageGenLoading && !previewLoading && !imageGenResult && !previewResult && (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', fontSize: '14px' }}>
                  Enter a prompt and press Generate or Preview
                </div>
              )}
            </div>

            {/* Prompt analysis + feedback */}
            {imageGenMeta && (
              <div style={{
                backgroundColor: 'rgba(10,10,20,0.95)', borderRadius: '14px',
                border: '1px solid rgba(179,136,255,0.2)', backdropFilter: 'blur(16px)',
                padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px',
              }}>
                {/* Prompt analysis */}
                <div>
                  <h4 style={{ margin: '0 0 6px', fontSize: '12px', color: '#b388ff' }}>Prompt Analysis</h4>
                  <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>
                    Seed: <span style={{ color: '#fff', fontFamily: 'monospace' }}>{imageGenMeta.seed}</span>
                    {imageGenMeta.settings && <> | Model: <span style={{ color: '#fff' }}>{imageGenMeta.settings.model}</span></>}
                    {imageGenMeta.long_prompt && (
                      <> | <span style={{ color: '#00cc88' }}>Long prompt ✓ (compel encoded)</span></>
                    )}
                  </div>
                  {imageGenMeta.prompt_analysis?.estimated_focus && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                      {imageGenMeta.prompt_analysis.estimated_focus.map((f: any, i: number) => (
                        <span key={i} style={{
                          padding: '2px 8px', borderRadius: '12px', fontSize: '10px',
                          backgroundColor: f.priority === 'high' ? 'rgba(0,204,136,0.2)' : f.priority === 'medium' ? 'rgba(255,153,0,0.15)' : 'rgba(100,100,100,0.15)',
                          color: f.priority === 'high' ? '#00cc88' : f.priority === 'medium' ? '#ff9900' : '#666',
                          border: '1px solid',
                          borderColor: f.priority === 'high' ? 'rgba(0,204,136,0.3)' : f.priority === 'medium' ? 'rgba(255,153,0,0.2)' : 'rgba(100,100,100,0.2)',
                        }}>
                          {f.text}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Full prompt used */}
                <details style={{ fontSize: '11px', color: '#888' }}>
                  <summary style={{ cursor: 'pointer', color: '#666' }}>Full prompt sent to model</summary>
                  <pre style={{ marginTop: '4px', padding: '8px', backgroundColor: '#0a0a0a', borderRadius: '6px', whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '10px', color: '#aaa', maxHeight: '120px', overflowY: 'auto' }}>
                    {imageGenMeta.prompt_used}
                  </pre>
                </details>

                {/* Feedback input */}
                <div style={{ display: 'flex', gap: '8px' }}>
                  <input
                    type="text"
                    value={imageGenFeedback}
                    onChange={e => setImageGenFeedback(e.target.value)}
                    placeholder="What went wrong? (e.g., bad hands, wrong position, wrong clothing...)"
                    style={{ flex: 1, padding: '7px 10px', borderRadius: '8px', border: '1px solid #333', backgroundColor: '#111', color: '#fff', fontSize: '12px', outline: 'none' }}
                  />
                  <button
                    disabled={!imageGenFeedback.trim()}
                    onClick={() => submitImageFeedback(imageGenHistory.length - 1, imageGenFeedback)}
                    style={{
                      padding: '7px 14px', borderRadius: '8px', border: 'none',
                      backgroundColor: imageGenFeedback.trim() ? '#ff9900' : '#333',
                      color: '#fff', cursor: imageGenFeedback.trim() ? 'pointer' : 'not-allowed',
                      fontSize: '12px', whiteSpace: 'nowrap',
                    }}
                  >
                    Submit Feedback
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right panel: History */}
          <div style={{
            width: '280px', flexShrink: 0,
            backgroundColor: 'rgba(10,10,20,0.95)', borderRadius: '14px',
            border: '1px solid rgba(179,136,255,0.2)', backdropFilter: 'blur(16px)',
            overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px',
          }}>
            <h4 style={{ margin: 0, fontSize: '13px', color: '#888', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              Recent Generations ({imageGenHistory.length})
              <button onClick={fetchImageHistory} style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: '12px' }}>⟳</button>
            </h4>
            {imageGenHistory.length === 0 && <div style={{ color: '#444', fontSize: '11px', textAlign: 'center', padding: '20px 0' }}>No history yet — generate an image to see it here</div>}
            {imageGenHistory.slice().reverse().map((h, i) => (
              <div key={i} style={{
                padding: '8px', borderRadius: '8px', backgroundColor: '#111',
                border: '1px solid #222', fontSize: '11px', cursor: 'pointer',
              }}
                onClick={() => {
                  let cleanPrompt = h.prompt?.replace(/^(score_9, score_8_up, score_7_up, |rating:\w+, )/, '') || '';
                  // Strip COMPEL/multi-character artifacts for cleaner reuse
                  cleanPrompt = cleanPrompt
                    .replace(/\s*BREAK\s*/g, ' ')
                    .replace(/,?\s*(?:on the (?:left|right)|in the (?:center|background))\s*/gi, ' ')
                    .replace(/\(:\d+\.?\d*\)/g, '')
                    .replace(/\(([^:()]+):\d+\.?\d*\)/g, '$1')
                    .replace(/,\s*,+/g, ',')
                    .replace(/\s{2,}/g, ' ')
                    .trim().replace(/^,|,$/g, '').trim();
                  setImageGenPrompt(cleanPrompt);
                  if (h.settings?.seed) setImageGenSeed(h.settings.seed);
                  if (h.settings?.steps) setImageGenSteps(h.settings.steps);
                  if (h.settings?.guidance_scale) setImageGenCfg(h.settings.guidance_scale);
                }}
              >
                {h.result_path && (
                  <img
                    src={`http://localhost:8000/file?path=${encodeURIComponent(h.result_path)}`}
                    alt=""
                    onClick={(e) => {
                      e.stopPropagation();
                      setLightboxUrl(`http://localhost:8000/file?path=${encodeURIComponent(h.result_path)}`);
                      setLightboxZoom(1);
                      setLightboxPan({ x: 0, y: 0 });
                    }}
                    style={{ width: '100%', height: '120px', objectFit: 'cover', borderRadius: '6px', marginBottom: '6px', cursor: 'zoom-in' }}
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                  />
                )}
                <div style={{ color: '#ccc', marginBottom: '4px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '11px' }}>
                  {h.prompt?.replace(/^(score_9, score_8_up, score_7_up, |rating:\w+, )/, '').slice(0, 80) || 'Unknown'}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#555', fontSize: '10px' }}>
                  <span>{h.settings?.model ? String(h.settings.model).split(/[/\\]/).pop()?.replace('.safetensors', '') : '?'}</span>
                  <span>seed:{h.settings?.seed ?? '?'}</span>
                </div>
                {h.timestamp && (
                  <div style={{ color: '#444', fontSize: '9px', marginTop: '2px' }}>
                    {new Date(h.timestamp).toLocaleString()}
                  </div>
                )}
                {h.feedback && (
                  <div style={{ marginTop: '4px', padding: '3px 6px', borderRadius: '4px', backgroundColor: 'rgba(255,153,0,0.1)', color: '#ff9900', fontSize: '10px' }}>
                    📝 {h.feedback.slice(0, 40)}
                  </div>
                )}
                {h.result_path && (
                  <div style={{ display: 'flex', gap: '4px', marginTop: '4px' }}>
                    <button onClick={(e) => {
                      e.stopPropagation();
                      fetch(`http://localhost:8000/file/open?path=${encodeURIComponent(h.result_path)}`, { method: 'POST' });
                    }} style={{ flex: 1, padding: '3px', borderRadius: '4px', border: '1px solid #333', backgroundColor: '#111', color: '#888', cursor: 'pointer', fontSize: '9px' }}
                      title="Open in system viewer">📂 Open</button>
                    <button onClick={(e) => {
                      e.stopPropagation();
                      setUpscaleImagePath(h.result_path);
                      setImageGenSubTab('upscale');
                    }} style={{ flex: 1, padding: '3px', borderRadius: '4px', border: '1px solid #333', backgroundColor: '#111', color: '#888', cursor: 'pointer', fontSize: '9px' }}
                      title="Send to upscale tab">🔍 Upscale</button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Chat history — infinite scroll */}
      {activeMode !== 'tutor' && activeMode !== 'imagegen' && (
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
            width: '90%',
            maxWidth: activeMode === 'agent' ? '1100px' : '960px',
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
            const modes: ActiveMode[] = ['query', 'agent', 'tutor', 'imagegen'];
            const idx = modes.indexOf(activeMode);
            setActiveMode(modes[(idx + 1) % modes.length]);
          }}
          disabled={mode !== 'idle'}
          style={{
            padding: '10px 14px', borderRadius: '12px', border: '2px solid',
            borderColor: activeMode === 'agent' ? '#ff00ff' : activeMode === 'tutor' ? '#00cc88' : activeMode === 'imagegen' ? '#b388ff' : '#5533ff',
            backgroundColor: activeMode === 'agent' ? 'rgba(255,0,255,0.1)' : activeMode === 'tutor' ? 'rgba(0,204,136,0.1)' : activeMode === 'imagegen' ? 'rgba(179,136,255,0.1)' : 'rgba(85,51,255,0.1)',
            color: activeMode === 'agent' ? '#ff00ff' : activeMode === 'tutor' ? '#00cc88' : activeMode === 'imagegen' ? '#b388ff' : '#5533ff',
            cursor: 'pointer', fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', transition: 'all 0.3s',
            backdropFilter: 'blur(10px)',
          }}
        >
          {activeMode === 'agent' ? '🤖 Agent' : activeMode === 'tutor' ? '📚 Tutor' : activeMode === 'imagegen' ? '🎨 ImageGen' : '🔍 Query'}
        </button>
      </div>

      {/* Input bar fixed at bottom center */}
      {activeMode !== 'tutor' && activeMode !== 'imagegen' && (
      <form
        onSubmit={handleSubmit}
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '88%',
          maxWidth: activeMode === 'agent' ? '900px' : '740px',
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
        {/* Agent mode dropdown — replaces separate Multi-Agent / Two-Mode toggles */}
        {activeMode === 'agent' && (
          <select
            value={useMultiAgent ? 'multi' : useTwoMode ? 'twomode' : 'single'}
            onChange={e => {
              const v = e.target.value;
              setUseMultiAgent(v === 'multi');
              setUseTwoMode(v === 'twomode');
            }}
            disabled={mode !== 'idle'}
            title="Agent orchestration mode"
            style={{
              padding: '8px 10px', borderRadius: '10px',
              border: '2px solid',
              borderColor: useMultiAgent ? '#ff00ff' : useTwoMode ? '#22aaff' : '#555',
              backgroundColor: useMultiAgent ? 'rgba(255,0,255,0.12)' : useTwoMode ? 'rgba(34,170,255,0.12)' : '#111',
              color: useMultiAgent ? '#ff00ff' : useTwoMode ? '#22aaff' : '#aaa',
              fontSize: '11px', cursor: 'pointer',
              transition: 'all 0.3s',
              appearance: 'none', WebkitAppearance: 'none',
              backgroundImage: 'url("data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%278%27 height=%275%27%3E%3Cpath d=%27M0 0l4 5 4-5z%27 fill=%27%23888%27/%3E%3C/svg%3E")',
              backgroundRepeat: 'no-repeat', backgroundPosition: 'right 8px center',
              paddingRight: '24px',
            }}
          >
            <option value="single">⚡ Single Agent</option>
            <option value="multi">🔗 Multi-Agent</option>
            <option value="twomode">🎯 Two-Mode</option>
          </select>
        )}

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

        {/* Persistent OCR context indicator */}
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

        <textarea
          value={input}
          onChange={e => { setInput(e.target.value); e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px'; }}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (input.trim() && mode === 'idle') { (e.target as HTMLTextAreaElement).form?.requestSubmit(); } } }}
          placeholder={pendingImageBlob ? 'Ask about the captured image...' : activeMode === 'agent' ? 'Describe code changes...' : 'Ask Aion...'}
          disabled={mode !== 'idle'}
          rows={1}
          style={{
            flex: 1,
            padding: '12px 18px',
            borderRadius: '18px',
            border: '1px solid #333',
            backgroundColor: '#111',
            color: '#fff',
            fontSize: '14px',
            outline: 'none',
            resize: 'none',
            overflow: 'hidden',
            lineHeight: '1.4',
            fontFamily: 'inherit',
            maxHeight: '150px',
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

      {/* ── Utility buttons — fixed bottom-left ────────────────────── */}
      {activeMode !== 'tutor' && activeMode !== 'imagegen' && (
      <div style={{
        position: 'fixed', bottom: '24px', left: '24px', zIndex: 45,
        display: 'flex', gap: '6px', alignItems: 'center',
        backgroundColor: 'rgba(10,10,15,0.85)', backdropFilter: 'blur(12px)',
        borderRadius: '14px', padding: '6px 10px',
        border: '1px solid rgba(255,255,255,0.08)',
      }}>
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
            padding: '8px 10px',
            borderRadius: '10px',
            border: '1px solid',
            borderColor: isRecording ? '#ff4444' : voiceLoading ? '#ff9900' : '#333',
            backgroundColor: isRecording ? 'rgba(255,68,68,0.2)' : voiceLoading ? 'rgba(255,153,0,0.15)' : 'transparent',
            color: isRecording ? '#ff4444' : voiceLoading ? '#ff9900' : '#888',
            cursor: mode !== 'idle' ? 'not-allowed' : 'pointer',
            fontSize: '15px',
            transition: 'all 0.15s',
            animation: isRecording ? 'pulse 0.8s infinite' : 'none',
          }}
        >
          {voiceLoading ? '⏳' : isRecording ? '⏹' : '🎤'}
        </button>

        {/* TTS — read last AI message aloud */}
        <button
          type="button"
            onClick={async () => {
              const lastAi = [...messages].reverse().find(m => m.role === 'ai');
              if (!lastAi) return;
              setTtsLoading(true);
              try {
                const res = await fetch('http://localhost:8000/voice/tts', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ text: lastAi.text.slice(0, 2000) }),
                });

                if (!res.ok) {
                  console.error('[TTS] backend error:', await res.text());
                  return;
                }

                const contentType = res.headers.get('content-type') ?? '';
                if (!contentType.includes('audio')) {
                  console.error('[TTS] wrong content-type, body:', await res.text());
                  return;
                }

                const blob = await res.blob();

                if (blob.size < 1000) {
                  console.error('[TTS] blob too small, likely an error response');
                  return;
                }

                if (ttsAudioRef.current) {
                  ttsAudioRef.current.pause();
                  URL.revokeObjectURL(ttsAudioRef.current.src);
                }

                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                ttsAudioRef.current = audio;

                try {
                  await audio.play();
                } catch (playErr) {
                  console.error('[TTS] audio.play() failed:', playErr);
                }

              } catch (err) {
                console.error('[TTS] fetch failed:', err);
              } finally {
                setTtsLoading(false);
              }
            }}
          disabled={ttsLoading || messages.filter(m => m.role === 'ai').length === 0}
          title="Read last AI response aloud (Kokoro TTS)"
          style={{
            padding: '8px 10px',
            borderRadius: '10px',
            border: '1px solid',
            borderColor: ttsLoading ? '#ff9900' : '#333',
            backgroundColor: ttsLoading ? 'rgba(255,153,0,0.15)' : 'transparent',
            color: ttsLoading ? '#ff9900' : '#888',
            cursor: ttsLoading ? 'wait' : 'pointer',
            fontSize: '15px',
            transition: 'all 0.15s',
          }}
        >
          {ttsLoading ? '⏳' : '🔊'}
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
              padding: '8px 10px',
              borderRadius: '10px',
              border: '1px solid',
              borderColor: ocrLoading ? '#ff9900' : showOcrMenu ? '#22bbff' : '#333',
              backgroundColor: ocrLoading ? 'rgba(255,153,0,0.15)' : showOcrMenu ? 'rgba(34,187,255,0.15)' : 'transparent',
              color: ocrLoading ? '#ff9900' : '#888',
              cursor: mode !== 'idle' || ocrLoading ? 'not-allowed' : 'pointer',
              fontSize: '15px',
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

        {/* Process log toggle */}
        <button
          type="button"
          onClick={() => setShowProcessPanel(p => !p)}
          title="Toggle backend process logs"
          style={{
            padding: '8px 10px',
            borderRadius: '10px',
            border: '1px solid',
            borderColor: showProcessPanel ? '#ff9900' : '#333',
            backgroundColor: showProcessPanel ? 'rgba(255,153,0,0.15)' : 'transparent',
            color: showProcessPanel ? '#ff9900' : '#888',
            cursor: 'pointer',
            fontSize: '13px',
            transition: 'all 0.15s',
            position: 'relative',
          }}
        >
          ⚙{processLogs.length > 0 && <span style={{ position: 'absolute', top: '-4px', right: '-4px', backgroundColor: '#ff9900', color: '#000', borderRadius: '50%', width: '14px', height: '14px', fontSize: '8px', fontWeight: 'bold', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{processLogs.length > 99 ? '99+' : processLogs.length}</span>}
        </button>
      </div>
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

      {/* ── Global Lightbox with zoom + pan ─────────────────────────── */}
      {lightboxUrl && (
        <div
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setLightboxUrl(null);
              setLightboxZoom(1);
              setLightboxPan({ x: 0, y: 0 });
            }
          }}
          onWheel={(e) => {
            e.preventDefault();
            setLightboxZoom(z => Math.max(0.5, Math.min(5, z + (e.deltaY < 0 ? 0.25 : -0.25))));
          }}
          style={{
            position: 'fixed', inset: 0, zIndex: 999,
            backgroundColor: 'rgba(0,0,0,0.85)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: lightboxZoom > 1 ? 'grab' : 'zoom-in',
            overflow: 'hidden',
          }}
        >
          <img
            src={lightboxUrl}
            alt="Enlarged"
            draggable={false}
            onMouseDown={(e) => {
              if (lightboxZoom > 1) {
                lightboxDragging.current = true;
                lightboxDragStart.current = { x: e.clientX - lightboxPan.x, y: e.clientY - lightboxPan.y };
                e.preventDefault();
              }
            }}
            onMouseMove={(e) => {
              if (lightboxDragging.current) {
                setLightboxPan({
                  x: e.clientX - lightboxDragStart.current.x,
                  y: e.clientY - lightboxDragStart.current.y,
                });
              }
            }}
            onMouseUp={() => { lightboxDragging.current = false; }}
            onMouseLeave={() => { lightboxDragging.current = false; }}
            onClick={(e) => {
              e.stopPropagation();
              if (lightboxZoom <= 1) {
                setLightboxZoom(2);
              }
            }}
            style={{
              maxWidth: '90vw', maxHeight: '90vh',
              borderRadius: '10px',
              border: '1px solid rgba(179,136,255,0.4)',
              boxShadow: '0 0 60px rgba(0,0,0,0.9)',
              transform: `scale(${lightboxZoom}) translate(${lightboxPan.x / lightboxZoom}px, ${lightboxPan.y / lightboxZoom}px)`,
              transition: lightboxDragging.current ? 'none' : 'transform 0.15s',
              cursor: lightboxZoom > 1 ? (lightboxDragging.current ? 'grabbing' : 'grab') : 'zoom-in',
              userSelect: 'none',
            }}
          />
          {/* Zoom controls */}
          <div style={{
            position: 'fixed', bottom: '20px', left: '50%', transform: 'translateX(-50%)',
            display: 'flex', gap: '8px', alignItems: 'center', zIndex: 1000,
            backgroundColor: 'rgba(0,0,0,0.7)', borderRadius: '20px', padding: '6px 14px',
          }}>
            <button onClick={(e) => { e.stopPropagation(); setLightboxZoom(z => Math.max(0.5, z - 0.25)); }}
              style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px', padding: '2px 6px' }}>−</button>
            <span style={{ color: '#aaa', fontSize: '12px', minWidth: '40px', textAlign: 'center' }}>{Math.round(lightboxZoom * 100)}%</span>
            <button onClick={(e) => { e.stopPropagation(); setLightboxZoom(z => Math.min(5, z + 0.25)); }}
              style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px', padding: '2px 6px' }}>+</button>
            <button onClick={(e) => { e.stopPropagation(); setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 }); }}
              style={{ background: 'none', border: '1px solid #555', borderRadius: '6px', color: '#888', cursor: 'pointer', fontSize: '10px', padding: '3px 8px', marginLeft: '4px' }}>Reset</button>
            <button onClick={(e) => { e.stopPropagation(); setLightboxUrl(null); setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 }); }}
              style={{ background: 'none', border: '1px solid #555', borderRadius: '6px', color: '#ff6666', cursor: 'pointer', fontSize: '10px', padding: '3px 8px' }}>Close</button>
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
@keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
`;
document.head.appendChild(styleEl);

