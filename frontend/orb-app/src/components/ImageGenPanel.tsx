import React, { useState, useRef, useEffect } from 'react';

type ImageModel = { name: string; filename: string; path: string; size_mb: number; type: 'checkpoint' | 'lora' };
type CharEntry = { lora: string; description: string; position: string; outfit: string };
type OrbMode = 'idle' | 'querying' | 'agent-processing';

interface ImageGenPanelProps {
  setMode: (mode: OrbMode) => void;
  onOpenLightbox: (url: string) => void;
}

export const ImageGenPanel: React.FC<ImageGenPanelProps> = ({ setMode, onOpenLightbox }) => {
  const [imageGenLoading, setImageGenLoading] = useState(false);
  const [imageGenPrompt, setImageGenPrompt] = useState('');
  const [imageGenResult, setImageGenResult] = useState<string | null>(null);
  const [showImageGenDialog, setShowImageGenDialog] = useState(false);
  // ── Image generation page state ────────────────────────────────────────────
  const [imageModels, setImageModels] = useState<ImageModel[]>([]);
  const [activeImageModel, setActiveImageModel] = useState<string | null>(null);
  const [selectedImageModel, setSelectedImageModel] = useState<string>('');
  const [selectedLoras, setSelectedLoras] = useState<string[]>(() => {
    try { const s = localStorage.getItem('aion_selectedLoras'); return s ? JSON.parse(s) : []; } catch { return []; }
  });
  const [loraWeights, setLoraWeights] = useState<Record<string, number>>(() => {
    try { const s = localStorage.getItem('aion_loraWeights'); return s ? JSON.parse(s) : {}; } catch { return {}; }
  });
  const [selectedOutfits, setSelectedOutfits] = useState<Record<string, string>>(() => {
    try { const s = localStorage.getItem('aion_selectedOutfits'); return s ? JSON.parse(s) : {}; } catch { return {}; }
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
  const [feedbackLearnings, setFeedbackLearnings] = useState<any>(null);
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
  // ── Multi-character composition panel ──────────────────────────────────────
  const [charEntries, setCharEntries] = useState<CharEntry[]>([]);
  const [charPanelOpen, setCharPanelOpen] = useState(true);
  const [charOutfitOptions, setCharOutfitOptions] = useState<Record<string, string[]>>({}); // lora -> outfit names
  const [charSharedPrompt, setCharSharedPrompt] = useState('');
  const [charEnvironment, setCharEnvironment] = useState('');
  const [charClothingText, setCharClothingText] = useState('');
  // ── UI layout ──────────────────────────────────────────────────────────────
  const [modelsExpanded, setModelsExpanded] = useState(false);
  // ── Character validation ───────────────────────────────────────────────────
  const [charValidation, setCharValidation] = useState<{ detected_characters: any[]; warnings: any[]; unmatched_loras: string[] } | null>(null);
  // ── History management ─────────────────────────────────────────────────────
  const [expandedHistoryIdx, setExpandedHistoryIdx] = useState<number | null>(null);
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
  const [loraCategories, setLoraCategories] = useState<Record<string, {name:string; filename:string; path:string; size_mb:number; trigger_words:string[]; preview_image?:string|null}[]>>({});
  const [loraBrowserOpen, setLoraBrowserOpen] = useState(false);
  const [loraBrowserCategory, setLoraBrowserCategory] = useState('styles');
  const [expandedLoraCategories, setExpandedLoraCategories] = useState<Record<string, boolean>>({ characters: true });
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

  const [vramFlushing, setVramFlushing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

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

  // ── Character validation (debounced) ─────────────────────────────────────
  useEffect(() => {
    if (!imageGenPrompt.trim()) { setCharValidation(null); return; }
    const timer = setTimeout(async () => {
      try {
        const loraStems = selectedLoras.map((l: string) => l.split(/[/\\]/).pop()?.replace('.safetensors', '') || l);
        const res = await fetch('http://localhost:8000/generate/validate-characters', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: imageGenPrompt, loras: loraStems }),
        });
        if (res.ok) setCharValidation(await res.json());
      } catch { /* backend down */ }
    }, 800);
    return () => clearTimeout(timer);
  }, [imageGenPrompt, selectedLoras]);

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

  // Fetch data on mount (component only renders when activeMode === 'imagegen')
  useEffect(() => {
    fetchImageModels(); fetchImageHistory(); fetchArtStyles(); fetchAnimJobs(); fetchLoraCategories();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist selectedLoras, loraWeights, selectedOutfits and artStyle to localStorage
  useEffect(() => {
    try { localStorage.setItem('aion_selectedLoras', JSON.stringify(selectedLoras)); } catch {}
  }, [selectedLoras]);
  useEffect(() => {
    try { localStorage.setItem('aion_loraWeights', JSON.stringify(loraWeights)); } catch {}
  }, [loraWeights]);
  useEffect(() => {
    try { localStorage.setItem('aion_selectedOutfits', JSON.stringify(selectedOutfits)); } catch {}
  }, [selectedOutfits]);
  useEffect(() => {
    try { localStorage.setItem('aion_imageGenArtStyle', imageGenArtStyle); } catch {}
  }, [imageGenArtStyle]);

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
    if (selectedLoras.length > 0) {
      body.loras = selectedLoras;
      body.lora_weights = selectedLoras.map(l => loraWeights[l] ?? 0.8);
      if (Object.keys(selectedOutfits).length > 0) body.selected_outfits = selectedOutfits;
    }
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
    if (selectedLoras.length > 0) {
      body.loras = selectedLoras;
      body.lora_weights = selectedLoras.map(l => loraWeights[l] ?? 0.8);
      if (Object.keys(selectedOutfits).length > 0) body.selected_outfits = selectedOutfits;
    }
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
    if (selectedLoras.length > 0) {
      body.loras = selectedLoras;
      // Send per-LoRA weights (default 0.8 if not set)
      body.lora_weights = selectedLoras.map(l => loraWeights[l] ?? 0.8);
      // Send selected outfits for multi-costume LoRAs
      if (Object.keys(selectedOutfits).length > 0) body.selected_outfits = selectedOutfits;
    }
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
    fetchFeedbackLearnings();
  } catch {}
};

// Fetch feedback learnings
const fetchFeedbackLearnings = async () => {
  try {
    const res = await fetch('http://localhost:8000/generate/feedback/learnings');
    const data = await res.json();
    if (data.status === 'ok') setFeedbackLearnings(data);
  } catch {}
};

// Clear feedback learnings
const clearFeedbackLearnings = async (scope: string) => {
  try {
    await fetch('http://localhost:8000/generate/feedback/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope }),
    });
    fetchFeedbackLearnings();
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

// Insert trigger word into prompt (with deduplication)
const insertTriggerWord = (word: string) => {
  setImageGenPrompt(prev => {
    const trimmed = prev.trim();
    // Check if the trigger word is already in the prompt (case-insensitive)
    const parts = trimmed.split(/,\s*/);
    if (parts.some(p => p.trim().toLowerCase() === word.toLowerCase())) return trimmed;
    return trimmed ? trimmed + ', ' + word : word;
  });
};

// Remove a trigger word from the prompt
const removeTriggerWord = (word: string) => {
  setImageGenPrompt(prev => {
    const parts = prev.split(/,\s*/);
    const filtered = parts.filter(p => p.trim().toLowerCase() !== word.toLowerCase());
    return filtered.join(', ');
  });
};

// Check if a LoRA belongs to the styles category (not a character)
const isStyleLora = (loraName: string): boolean => {
  const styleLoras = loraCategories['styles'] || [];
  return styleLoras.some((lr: any) => lr.name === loraName);
};

// Toggle a LoRA and auto-insert/remove its primary trigger word
// Style LoRAs skip prompt injection — they work via LoRA weights only
const toggleLoraWithTrigger = (loraName: string, triggerWords?: string[]) => {
  setSelectedLoras(prev => {
    const isActive = prev.includes(loraName);
    const isStyle = isStyleLora(loraName);
    if (isActive) {
      // Deselecting: remove primary trigger word from prompt (skip for styles)
      if (triggerWords?.length && !isStyle) removeTriggerWord(triggerWords[0]);
      return prev.filter(x => x !== loraName);
    } else {
      // Selecting: auto-insert primary trigger word (skip for styles)
      if (triggerWords?.length && !isStyle) insertTriggerWord(triggerWords[0]);
      return [...prev, loraName];
    }
  });
};

// ── Multi-character composition helpers ──────────────────────────────────
const getCharacterLoras = (): string[] => {
  const charLoras = (loraCategories['characters'] || []) as any[];
  return selectedLoras.filter(l => charLoras.some((lr: any) => lr.name === l));
};

// Fetch outfit options for a character LoRA
const fetchCharOutfits = async (loraName: string) => {
  try {
    const res = await fetch(`http://localhost:8000/generate/loras/trigger-words/parsed/${encodeURIComponent(loraName)}`);
    const data = await res.json();
    if (data.has_outfits && data.outfits) {
      setCharOutfitOptions(prev => ({ ...prev, [loraName]: Object.keys(data.outfits) }));
    }
  } catch {}
};

// Auto-sync character entries when selectedLoras changes
useEffect(() => {
  const charLoras = getCharacterLoras();
  if (charLoras.length < 2) {
    // Not multi-char mode
    if (charEntries.length > 0) setCharEntries([]);
    return;
  }
  const positions = ['on the left', 'on the right', 'in the center', 'in the background'];
  setCharEntries(prev => {
    const updated: CharEntry[] = [];
    charLoras.forEach((l, i) => {
      const existing = prev.find(e => e.lora === l);
      if (existing) {
        updated.push(existing);
      } else {
        // Find the LoRA data to get primary trigger
        const loraData = (loraCategories['characters'] as any[] || []).find((lr: any) => lr.name === l);
        const tw = loraData?.trigger_words || [];
        // Get primary trigger: first word (or first part before colon)
        let primaryTrigger = l;
        if (tw.length > 0) {
          const first = tw[0];
          const colonMatch = first.match(/^(.+?):\s*(.+)/);
          if (colonMatch) {
            // Colon format: use the trigger code from first entry
            const rest = colonMatch[1].trim();
            primaryTrigger = rest.includes('Base') ? (tw[0]) : first;
          } else {
            primaryTrigger = first;
          }
        }
        updated.push({ lora: l, description: '', position: positions[i] || positions[positions.length - 1], outfit: '' });
        // Fetch outfit options
        fetchCharOutfits(l);
      }
    });
    return updated;
  });
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [selectedLoras, loraCategories]);

// Compose BREAK-separated prompt from character panel
const composeCharacterPrompt = () => {
  const charLoras = getCharacterLoras();
  if (charLoras.length < 2 || charEntries.length < 2) return;

  const parts: string[] = [];

  // Shared prompt (scene description + environment)
  const shared = charSharedPrompt.trim();
  const env = charEnvironment.trim();
  const clothText = charClothingText.trim();

  let sharedBlock = `${charEntries.length} characters, group shot`;
  if (shared) sharedBlock += `, ${shared}`;
  if (env) sharedBlock += `, ${env}`;
  if (clothText) sharedBlock += `, ${clothText}`;

  parts.push(sharedBlock);

  // Each character section
  for (const entry of charEntries) {
    const loraData = (loraCategories['characters'] as any[] || []).find((lr: any) => lr.name === entry.lora);
    const tw = loraData?.trigger_words || [];

    let charBlock = '';

    if (entry.description.trim()) {
      // User has written a custom description
      charBlock = entry.description.trim();
    } else if (entry.outfit && tw.length > 0) {
      // User selected an outfit — find the tags for it
      for (const t of tw) {
        if (t.toLowerCase().startsWith(entry.outfit.toLowerCase())) {
          // Colon format: "Outfit Name:trigger, tag1, tag2"
          const colonIdx = t.indexOf(':');
          charBlock = colonIdx >= 0 ? t.substring(colonIdx + 1).trim() : t;
          break;
        }
      }
      if (!charBlock) charBlock = tw[0]; // fallback
    } else if (tw.length > 0) {
      // Use primary trigger as default
      const first = tw[0];
      const colonMatch = first.match(/^[^:]+:\s*(.+)/);
      charBlock = colonMatch ? colonMatch[1].trim() : first;
    } else {
      charBlock = entry.lora;
    }

    // Wrap in parentheses with weight if not already wrapped
    if (!charBlock.startsWith('(')) {
      // Extract trigger code (first comma-separated part) and give it weight
      const firstComma = charBlock.indexOf(',');
      if (firstComma > 0) {
        const trigger = charBlock.substring(0, firstComma).trim();
        const rest = charBlock.substring(firstComma + 1).trim();
        charBlock = `(${trigger}:1, ${rest})`;
      } else {
        charBlock = `(${charBlock}:1)`;
      }
    }

    parts.push(`${charBlock}, ${entry.position}`);
  }

  // Set the composed prompt
  setImageGenPrompt(parts.join(' BREAK '));

  // Also set outfits for backend
  const outfitMap: Record<string, string> = {};
  for (const entry of charEntries) {
    if (entry.outfit) outfitMap[entry.lora] = entry.outfit;
  }
  setSelectedOutfits(outfitMap);
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
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  } catch {}
  setImageGenLoading(false);
  setMode('idle');
};

// Free all VRAM (unload SD pipelines + evict Ollama models)
  const handleFlushVram = async () => {
    setVramFlushing(true);
    try {
      await fetch('http://localhost:8000/generate/vram/flush', { method: 'POST' });
    } catch {}
    setVramFlushing(false);
  };

  return (
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
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1, minWidth: 0 }}>
                        {lr.preview_image && (
                          <img src={`http://localhost:8000/generate/lora-preview/${encodeURIComponent(lr.name)}`} alt="" style={{ width: '32px', height: '32px', borderRadius: '4px', objectFit: 'cover', flexShrink: 0 }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
                        )}
                        <div style={{ minWidth: 0 }}>
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
                      </div>
                      <button onClick={() => toggleLoraWithTrigger(lr.name, lr.trigger_words)}
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

          {/* Selected LoRAs chips with weight sliders */}
          {selectedLoras.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '6px' }}>
              {selectedLoras.map(l => {
                // Find trigger words for this LoRA from loaded categories
                const loraData = Object.values(loraCategories).flat().find((lr: any) => lr.name === l) as any;
                const loraTw = loraData?.trigger_words;
                const weight = loraWeights[l] ?? 0.8;
                // Check if this LoRA has outfit groups:
                // 1) Semicolon-delimited: ["trigger", ";", "outfitName", ...]
                // 2) Colon-delimited: ["OutfitName:TriggerCode, tag1, tag2", ...]
                const hasOutfitsSemicolon = loraTw && loraTw.some((tw: string) => tw === ';');
                const hasOutfitsColon = loraTw && !hasOutfitsSemicolon && loraTw.filter((tw: string) => /^.+:\s*.+/.test(tw)).length >= 2;
                const hasOutfits = hasOutfitsSemicolon || hasOutfitsColon;
                let outfitNames: string[] = [];
                if (hasOutfitsSemicolon) {
                  let inGroup = false;
                  for (const tw of loraTw) {
                    if (tw === ';') { inGroup = true; continue; }
                    if (inGroup) { outfitNames.push(tw); inGroup = false; }
                  }
                } else if (hasOutfitsColon) {
                  for (const tw of loraTw) {
                    const m = tw.match(/^(.+?):\s*.+/);
                    if (m) outfitNames.push(m[1].trim());
                  }
                }
                return (
                <div key={l} style={{
                  padding: '4px 8px', borderRadius: '10px', fontSize: '10px',
                  backgroundColor: 'rgba(179,136,255,0.15)', color: '#b388ff', border: '1px solid rgba(179,136,255,0.3)',
                  display: 'flex', flexDirection: 'column', gap: '3px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{l}</span>
                    {/* Weight slider */}
                    <input type="range" min="0" max="1.5" step="0.05" value={weight}
                      onChange={e => setLoraWeights(prev => ({ ...prev, [l]: parseFloat(e.target.value) }))}
                      style={{ width: '60px', height: '12px', accentColor: '#b388ff' }}
                      title={`Weight: ${weight.toFixed(2)}`} />
                    <span style={{ fontSize: '9px', color: '#888', minWidth: '28px', textAlign: 'right', fontFamily: 'monospace' }}>{weight.toFixed(2)}</span>
                    <span onClick={() => { if (loraTw?.length) removeTriggerWord(loraTw[0]); setSelectedLoras(prev => prev.filter(x => x !== l)); setLoraWeights(prev => { const n = {...prev}; delete n[l]; return n; }); }} style={{ cursor: 'pointer', color: '#ff4444', fontWeight: 'bold' }}>×</span>
                  </div>
                  {/* Outfit selector for multi-costume LoRAs */}
                  {outfitNames.length > 0 && (
                    <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap', paddingLeft: '4px' }}>
                      <span style={{ fontSize: '9px', color: '#666' }}>Outfit:</span>
                      {outfitNames.map(outfit => (
                        <span key={outfit}
                          onClick={() => setSelectedOutfits(prev => ({ ...prev, [l]: outfit }))}
                          style={{
                            padding: '1px 6px', borderRadius: '8px', fontSize: '9px', cursor: 'pointer',
                            backgroundColor: selectedOutfits[l] === outfit ? 'rgba(179,136,255,0.3)' : 'rgba(100,100,100,0.15)',
                            color: selectedOutfits[l] === outfit ? '#b388ff' : '#666',
                            border: `1px solid ${selectedOutfits[l] === outfit ? 'rgba(179,136,255,0.4)' : 'rgba(100,100,100,0.2)'}`,
                          }}>{outfit}</span>
                      ))}
                    </div>
                  )}
                </div>
                );
              })}
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
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1, minWidth: 0 }}>
                        {lr.preview_image && (
                          <img src={`http://localhost:8000/generate/lora-preview/${encodeURIComponent(lr.name)}`} alt="" style={{ width: '32px', height: '32px', borderRadius: '4px', objectFit: 'cover', flexShrink: 0 }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
                        )}
                        <div style={{ minWidth: 0 }}>
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
                      </div>
                      <button onClick={() => toggleLoraWithTrigger(lr.name, lr.trigger_words)}
                        style={{ background: 'none', border: '1px solid', borderColor: selectedLoras.includes(lr.name) ? '#b388ff' : '#444', borderRadius: '4px', color: selectedLoras.includes(lr.name) ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '9px', padding: '1px 6px' }}>
                        {selectedLoras.includes(lr.name) ? '✓' : '+'}
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Collapsible category sections */}
              {['characters', 'styles', 'clothing', 'poses', 'concept', 'action'].map(cat => {
                const items = loraCategories[cat] || [];
                const isExpanded = expandedLoraCategories[cat] ?? false;
                return (
                  <div key={cat} style={{ marginBottom: '4px' }}>
                    <button onClick={() => setExpandedLoraCategories(prev => ({ ...prev, [cat]: !prev[cat] }))}
                      style={{
                        width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '6px 8px', borderRadius: '6px', border: '1px solid #333', cursor: 'pointer',
                        backgroundColor: isExpanded ? 'rgba(179,136,255,0.1)' : '#0d0d0d',
                        color: isExpanded ? '#b388ff' : '#888', fontSize: '11px', textTransform: 'capitalize',
                      }}>
                      <span>{isExpanded ? '▼' : '▶'} {cat}</span>
                      <span style={{ fontSize: '9px', color: '#555' }}>{items.length}</span>
                    </button>
                    {isExpanded && items.length === 0 && (
                      <div style={{ color: '#444', fontSize: '10px', textAlign: 'center', padding: '8px 0' }}>No LoRAs in {cat}/</div>
                    )}
                    {isExpanded && items.map(lora => (
                <div key={lora.name} style={{
                  padding: '6px', borderRadius: '6px', backgroundColor: '#111', border: '1px solid #222',
                  marginBottom: '4px', marginTop: '4px', marginLeft: '4px', fontSize: '11px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1, minWidth: 0 }}>
                      {lora.preview_image && (
                        <img src={`http://localhost:8000/generate/lora-preview/${encodeURIComponent(lora.name)}`} alt=""
                          style={{ width: '36px', height: '36px', borderRadius: '4px', objectFit: 'cover', flexShrink: 0, border: '1px solid #333' }}
                          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
                      )}
                      <span style={{ color: '#ccc', fontWeight: 500 }}>{lora.name}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                      <span style={{ color: '#555', fontSize: '9px' }}>{lora.size_mb}MB</span>
                      <button onClick={() => toggleLoraWithTrigger(lora.name, lora.trigger_words)}
                        style={{ background: 'none', border: '1px solid', borderColor: selectedLoras.includes(lora.name) ? '#b388ff' : '#444', borderRadius: '4px', color: selectedLoras.includes(lora.name) ? '#b388ff' : '#888', cursor: 'pointer', fontSize: '9px', padding: '1px 6px' }}>
                        {selectedLoras.includes(lora.name) ? '✓ Active' : '+ Use'}
                      </button>
                    </div>
                  </div>

                  {/* Trigger words display */}
                  {lora.trigger_words?.length > 0 && editingTriggerLora !== lora.name && (
                    <div style={{ marginTop: '4px' }}>
                      {lora.trigger_words.length > 5 ? (
                        <details>
                          <summary style={{ fontSize: '9px', color: '#666', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <span>Triggers ({lora.trigger_words.length})</span>
                            {/* Show first 2 trigger words as preview */}
                            <span style={{ display: 'inline-flex', gap: '3px', marginLeft: '2px' }}>
                              {lora.trigger_words.slice(0, 2).map((tw: string, j: number) => (
                                <span key={j} style={{
                                  padding: '1px 5px', borderRadius: '8px', fontSize: '8px',
                                  backgroundColor: 'rgba(0,204,136,0.08)', color: '#00aa77',
                                  border: '1px solid rgba(0,204,136,0.15)',
                                }}>{tw.length > 20 ? tw.slice(0, 20) + '…' : tw}</span>
                              ))}
                              <span style={{ fontSize: '8px', color: '#555' }}>+{lora.trigger_words.length - 2}</span>
                            </span>
                          </summary>
                          <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '3px', alignItems: 'center' }}>
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
                        </details>
                      ) : (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px', alignItems: 'center' }}>
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
                );
              })}
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
            abortRef.current = controller;
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
              abortRef.current = null;
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
                    abortRef.current = controller;
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
                      abortRef.current = null;
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
       <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto', overflowX: 'hidden', minHeight: 0 }}>

        {/* Multi-character composition panel */}
        {charEntries.length >= 2 && (
          <div style={{
            backgroundColor: 'rgba(10,10,30,0.95)', borderRadius: '14px',
            border: '1px solid rgba(124,77,255,0.3)', backdropFilter: 'blur(16px)',
            padding: '12px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: charPanelOpen ? '10px' : '0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }} onClick={() => setCharPanelOpen(!charPanelOpen)}>
                <span style={{ fontSize: '12px', color: '#7c4dff', fontWeight: 600 }}>
                  {charPanelOpen ? '▼' : '▶'} Character Composer ({charEntries.length} characters)
                </span>
                <span style={{ fontSize: '9px', color: '#555' }}>Build BREAK-separated prompts per character</span>
              </div>
              <button onClick={composeCharacterPrompt} style={{
                padding: '5px 14px', borderRadius: '8px', border: 'none',
                background: 'linear-gradient(135deg, #7c4dff, #b388ff)', color: '#fff',
                fontSize: '11px', fontWeight: 'bold', cursor: 'pointer',
              }}>⚡ Compose Prompt</button>
            </div>

            {charPanelOpen && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {/* Shared scene description */}
                <div style={{ padding: '8px', borderRadius: '8px', backgroundColor: 'rgba(0,204,136,0.05)', border: '1px solid rgba(0,204,136,0.15)' }}>
                  <label style={{ fontSize: '10px', color: '#00cc88', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Scene Description (shared between all characters)
                  </label>
                  <input type="text" value={charSharedPrompt}
                    onChange={e => setCharSharedPrompt(e.target.value)}
                    placeholder="animescreencap, stretching, dynamic angle, cinematic lighting..."
                    style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#0a0a0a', color: '#fff', fontSize: '11px', outline: 'none', boxSizing: 'border-box' }} />
                </div>

                {/* Environment & clothing text row */}
                <div style={{ display: 'flex', gap: '8px' }}>
                  <div style={{ flex: 1, padding: '8px', borderRadius: '8px', backgroundColor: 'rgba(68,170,255,0.05)', border: '1px solid rgba(68,170,255,0.15)' }}>
                    <label style={{ fontSize: '10px', color: '#44aaff', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                      🌍 Environment / Background
                    </label>
                    <input type="text" value={charEnvironment}
                      onChange={e => setCharEnvironment(e.target.value)}
                      placeholder="bedroom, dim lighting, silk sheets, window with moonlight..."
                      style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#0a0a0a', color: '#fff', fontSize: '11px', outline: 'none', boxSizing: 'border-box' }} />
                  </div>
                  <div style={{ flex: 1, padding: '8px', borderRadius: '8px', backgroundColor: 'rgba(255,153,0,0.05)', border: '1px solid rgba(255,153,0,0.15)' }}>
                    <label style={{ fontSize: '10px', color: '#ff9900', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                      ✍ Clothing Text (if any has writing)
                    </label>
                    <input type="text" value={charClothingText}
                      onChange={e => setCharClothingText(e.target.value)}
                      placeholder='e.g. shirt says "LOVE", hat text "NYC"...'
                      style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#0a0a0a', color: '#fff', fontSize: '11px', outline: 'none', boxSizing: 'border-box' }} />
                  </div>
                </div>

                {/* Per-character entries */}
                {charEntries.map((entry, idx) => {
                  const loraData = (loraCategories['characters'] as any[] || []).find((lr: any) => lr.name === entry.lora);
                  const tw = loraData?.trigger_words || [];
                  const outfits = charOutfitOptions[entry.lora] || [];
                  // Detect colon-format outfit names from trigger words directly
                  const colonOutfits = tw.filter((t: string) => /^.+:\s*.+/.test(t)).map((t: string) => {
                    const m = t.match(/^(.+?):\s*/);
                    return m ? m[1].trim() : t;
                  });
                  const allOutfits = outfits.length > 0 ? outfits : colonOutfits;

                  return (
                    <div key={entry.lora} style={{
                      padding: '8px', borderRadius: '8px',
                      backgroundColor: 'rgba(124,77,255,0.05)',
                      border: '1px solid rgba(124,77,255,0.2)',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <span style={{ fontSize: '11px', color: '#b388ff', fontWeight: 600 }}>
                          Character {idx + 1}: {entry.lora}
                        </span>
                        <select value={entry.position}
                          onChange={e => setCharEntries(prev => prev.map((c, i) => i === idx ? { ...c, position: e.target.value } : c))}
                          style={{ padding: '2px 8px', borderRadius: '6px', border: '1px solid #444', backgroundColor: '#111', color: '#fff', fontSize: '10px' }}>
                          <option value="on the left">Left</option>
                          <option value="on the right">Right</option>
                          <option value="in the center">Center</option>
                          <option value="in the background">Background</option>
                        </select>
                      </div>

                      {/* Outfit selector */}
                      {allOutfits.length > 0 && (
                        <div style={{ marginBottom: '6px' }}>
                          <label style={{ fontSize: '9px', color: '#666', display: 'block', marginBottom: '3px' }}>Outfit:</label>
                          <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                            {allOutfits.map((outfit: string) => (
                              <span key={outfit}
                                onClick={() => {
                                  setCharEntries(prev => prev.map((c, i) => i === idx ? { ...c, outfit: c.outfit === outfit ? '' : outfit } : c));
                                  // When outfit selected, auto-fill description
                                  const matchingTw = tw.find((t: string) => t.toLowerCase().startsWith(outfit.toLowerCase()));
                                  if (matchingTw) {
                                    const colonIdx = matchingTw.indexOf(':');
                                    const tags = colonIdx >= 0 ? matchingTw.substring(colonIdx + 1).trim() : matchingTw;
                                    setCharEntries(prev => prev.map((c, i) => i === idx ? { ...c, description: tags, outfit: c.outfit === outfit ? '' : outfit } : c));
                                  }
                                }}
                                style={{
                                  padding: '2px 8px', borderRadius: '8px', fontSize: '9px', cursor: 'pointer',
                                  backgroundColor: entry.outfit === outfit ? 'rgba(124,77,255,0.3)' : 'rgba(50,50,50,0.3)',
                                  color: entry.outfit === outfit ? '#b388ff' : '#888',
                                  border: `1px solid ${entry.outfit === outfit ? 'rgba(124,77,255,0.4)' : '#333'}`,
                                  maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                }}
                                title={outfit}
                              >{outfit.length > 30 ? outfit.slice(0, 30) + '…' : outfit}</span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Character description */}
                      <textarea
                        value={entry.description}
                        onChange={e => setCharEntries(prev => prev.map((c, i) => i === idx ? { ...c, description: e.target.value } : c))}
                        placeholder={`Character tags for ${entry.lora}... (e.g., GigiBase, pink eyes, multicolored hair, short twintails)`}
                        rows={2}
                        style={{
                          width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid #333',
                          backgroundColor: '#0a0a0a', color: '#eee', fontSize: '11px', outline: 'none',
                          resize: 'vertical', fontFamily: 'inherit', boxSizing: 'border-box',
                        }}
                      />

                      {/* Quick-insert trigger tags */}
                      {tw.length > 0 && (
                        <details style={{ marginTop: '4px' }}>
                          <summary style={{ fontSize: '9px', color: '#555', cursor: 'pointer' }}>Insert trigger tags</summary>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px', marginTop: '4px' }}>
                            {tw.map((t: string, j: number) => {
                              // For colon-format, show just the label as the tag
                              const isColonFmt = /^.+:\s*.+/.test(t);
                              const label = isColonFmt ? t.match(/^(.+?):/)?.[1]?.trim() || t : t;
                              return (
                                <span key={j}
                                  onClick={() => {
                                    const value = isColonFmt ? (t.substring(t.indexOf(':') + 1).trim()) : t;
                                    setCharEntries(prev => prev.map((c, i) => {
                                      if (i !== idx) return c;
                                      const existing = c.description.trim();
                                      return { ...c, description: existing ? existing + ', ' + value : value };
                                    }));
                                  }}
                                  style={{
                                    padding: '1px 6px', borderRadius: '8px', fontSize: '8px', cursor: 'pointer',
                                    backgroundColor: 'rgba(0,204,136,0.08)', color: '#00aa77',
                                    border: '1px solid rgba(0,204,136,0.15)',
                                  }}
                                  title={t}
                                >{label.length > 25 ? label.slice(0, 25) + '…' : label}</span>
                              );
                            })}
                          </div>
                        </details>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

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
              backgroundColor: '#0a0a14',
              border: `1px solid ${genProgress.message?.includes('Complete') ? 'rgba(0,204,136,0.5)' : genProgress.message?.includes('Saving') ? 'rgba(255,153,0,0.4)' : '#b388ff33'}`,
              fontSize: '10px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                <span style={{ color: genProgress.message?.includes('Complete') ? '#00cc88' : '#b388ff' }}>
                  {genProgress.message?.includes('Complete') ? '✓ Complete!'
                    : genProgress.message?.includes('Saving') ? '💾 Saving...'
                    : genProgress.type === 'animation'
                    ? `🎬 Frame ${genProgress.current_frame + 1}/${genProgress.total_frames}`
                    : genProgress.type === 'video' ? '🎥 Video'
                    : '🎨 Generating'}
                </span>
                <span style={{ color: '#aaa' }}>
                  Step {genProgress.current_step}/{genProgress.total_steps}
                  {genProgress.total_steps > 0 && ` (${Math.round(genProgress.current_step / genProgress.total_steps * 100)}%)`}
                  {genProgress.message && !genProgress.message.includes('Complete') && !genProgress.message.includes('Saving') && <> | <span style={{ color: '#00cc88' }}>{genProgress.message}</span></>}
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
                    height: '100%', borderRadius: '2px', transition: 'width 0.3s, background 0.3s',
                    width: `${genProgress.total_steps > 0 ? (genProgress.current_step / genProgress.total_steps * 100) : 0}%`,
                    background: genProgress.message?.includes('Complete') ? '#00cc88' : genProgress.message?.includes('Saving') ? '#ff9900' : '#7c4dff',
                  }} />
                )}
              </div>
              {genProgress.message && !genProgress.message.includes('Complete') && !genProgress.message.includes('Saving') && (
                <div style={{ color: '#666', marginTop: '2px', fontSize: '9px' }}>{genProgress.message}</div>
              )}
            </div>
          )}
        </div>

        {/* Result area */}
        <div style={{
          height: '450px', backgroundColor: 'rgba(10,10,20,0.95)', borderRadius: '14px',
          border: '1px solid rgba(179,136,255,0.2)', backdropFilter: 'blur(16px)',
          overflow: 'hidden', display: 'flex', flexDirection: 'column', flexShrink: 0,
          position: 'sticky', top: '0', zIndex: 5,
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
                onClick={() => { onOpenLightbox(previewResult); }}
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
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px', position: 'relative', height: '100%' }}>
                <img
                  src={imageGenResult}
                  alt="Generated"
                  onClick={() => { onOpenLightbox(imageGenResult); }}
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
        {/* Character validation warnings */}
        {charValidation && (charValidation.warnings?.length > 0 || charValidation.detected_characters?.length > 0) && (
          <div style={{
            backgroundColor: 'rgba(10,10,20,0.95)', borderRadius: '14px',
            border: `1px solid ${charValidation.warnings?.length > 0 ? 'rgba(255,100,100,0.4)' : 'rgba(0,204,136,0.3)'}`,
            padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: '6px',
          }}>
            <h4 style={{ margin: 0, fontSize: '11px', color: charValidation.warnings?.length > 0 ? '#ff6666' : '#00cc88' }}>
              {charValidation.warnings?.length > 0 ? '⚠ Character Detection' : '✓ Characters Detected'}
            </h4>
            {charValidation.detected_characters?.map((c: any, i: number) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
                <span style={{ padding: '2px 8px', borderRadius: '8px', backgroundColor: 'rgba(0,204,136,0.15)', color: '#00cc88', border: '1px solid rgba(0,204,136,0.3)' }}>
                  ✓ {c.trigger_code}
                </span>
                <span style={{ color: '#666' }}>→ LoRA: {c.lora}</span>
              </div>
            ))}
            {charValidation.warnings?.map((w: any, i: number) => (
              <div key={i} style={{ fontSize: '11px', padding: '4px 8px', borderRadius: '6px', backgroundColor: 'rgba(255,100,100,0.08)', border: '1px solid rgba(255,100,100,0.2)' }}>
                <span style={{ color: '#ff9900' }}>⚠ "{w.text}"</span>
                <span style={{ color: '#888' }}> — {w.reason}</span>
                <button onClick={() => {
                  setImageGenPrompt(prev => prev.replace(new RegExp(w.text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), w.suggestion));
                }} style={{ marginLeft: '8px', padding: '1px 8px', borderRadius: '4px', border: '1px solid rgba(0,204,136,0.3)', backgroundColor: 'rgba(0,204,136,0.1)', color: '#00cc88', cursor: 'pointer', fontSize: '10px' }}>
                  Fix → {w.suggestion}
                </button>
              </div>
            ))}
            {charValidation.unmatched_loras?.length > 0 && (
              <div style={{ fontSize: '10px', color: '#888' }}>
                ⚠ Loaded LoRAs not found in prompt: {charValidation.unmatched_loras.join(', ')}
              </div>
            )}
          </div>
        )}
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
                {imageGenMeta.prompt_analysis?.break_count > 0 && (
                  <> | <span style={{ color: '#7c4dff' }}>Auto-chunked ({imageGenMeta.prompt_analysis.break_count + 1} segments)</span></>
                )}
                {imageGenMeta.regional_mode && (
                  <> | <span style={{ color: '#ff6e40' }}>Regional conditioning ✓ (per-character isolation)</span></>
                )}
              </div>
              {imageGenMeta.prompt_analysis?.estimated_focus && (() => {
                // Regional layout info
                const layout = imageGenMeta.prompt_analysis?.regional_layout;

                // Group tags by section for collapsible character rendering
                const groups: { section: string; items: any[] }[] = [];
                let currentSection = '';
                imageGenMeta.prompt_analysis.estimated_focus.forEach((f: any) => {
                  const sec = f.section || 'shared';
                  if (sec !== currentSection) {
                    groups.push({ section: sec, items: [] });
                    currentSection = sec;
                  }
                  groups[groups.length - 1].items.push(f);
                });

                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {/* Regional layout visualization */}
                    {layout && layout.length >= 2 && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', padding: '4px 8px', borderRadius: '6px', backgroundColor: 'rgba(255,110,64,0.1)', border: '1px solid rgba(255,110,64,0.3)', marginBottom: '4px' }}>
                        <span style={{ fontSize: '9px', color: '#ff6e40', fontWeight: 600 }}>LAYOUT:</span>
                        <div style={{ display: 'flex', gap: '2px', flex: 1 }}>
                          {layout.map((r: any, i: number) => (
                            <div key={i} style={{
                              flex: r.position === 'background' ? undefined : 1,
                              padding: '2px 6px', borderRadius: '4px', textAlign: 'center' as const,
                              fontSize: '9px', fontWeight: 500,
                              backgroundColor: i === 0 ? 'rgba(0,176,255,0.2)' : i === 1 ? 'rgba(124,77,255,0.2)' : 'rgba(255,110,64,0.2)',
                              color: i === 0 ? '#00b0ff' : i === 1 ? '#b388ff' : '#ff6e40',
                              border: '1px solid',
                              borderColor: i === 0 ? 'rgba(0,176,255,0.4)' : i === 1 ? 'rgba(124,77,255,0.4)' : 'rgba(255,110,64,0.4)',
                            }}>
                              Char {r.region} → {r.position}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {groups.map((group, gi) => {
                      const isChar = group.section !== 'shared';
                      const sectionLabel = group.section === 'character_inline'
                        ? 'Character (inline — add BREAK for better isolation)'
                        : group.section.replace('_', ' ');
                      const tagElements = group.items.map((f: any, i: number) => (
                        <span key={i} style={{
                          padding: '2px 8px', borderRadius: '12px', fontSize: '10px',
                          backgroundColor: f.priority === 'high' ? 'rgba(0,204,136,0.2)' : f.priority === 'character' ? 'rgba(124,77,255,0.2)' : f.priority === 'medium' ? 'rgba(255,153,0,0.15)' : 'rgba(100,100,100,0.15)',
                          color: f.priority === 'high' ? '#00cc88' : f.priority === 'character' ? '#b388ff' : f.priority === 'medium' ? '#ff9900' : '#666',
                          border: '1px solid',
                          borderColor: f.priority === 'high' ? 'rgba(0,204,136,0.3)' : f.priority === 'character' ? 'rgba(124,77,255,0.3)' : f.priority === 'medium' ? 'rgba(255,153,0,0.2)' : 'rgba(100,100,100,0.2)',
                        }}>
                          {f.text}
                        </span>
                      ));

                      if (isChar) {
                        return (
                          <details key={gi} open>
                            <summary style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', margin: '2px 0' }}>
                              <span style={{ fontSize: '9px', color: group.section === 'character_inline' ? '#ff9900' : '#7c4dff', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                {group.section === 'character_inline' ? '⚠ ' : 'BREAK — '}{sectionLabel}
                              </span>
                              <span style={{ fontSize: '9px', color: '#555' }}>({group.items.length} tags)</span>
                              <div style={{ flex: 1, height: '1px', background: 'rgba(124,77,255,0.3)' }} />
                            </summary>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px', paddingLeft: '8px' }}>
                              {tagElements}
                            </div>
                          </details>
                        );
                      }
                      return (
                        <div key={gi} style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                          {tagElements}
                        </div>
                      );
                    })}
                  </div>
                );
              })()}
            </div>

            {/* LoRA diagnostics */}
            {imageGenMeta.lora_diagnostics?.length > 0 && (
              <details style={{ fontSize: '11px', color: '#888' }}>
                <summary style={{ cursor: 'pointer', color: '#b388ff' }}>LoRA Status ({imageGenMeta.lora_diagnostics.filter((d: any) => d.loaded).length}/{imageGenMeta.lora_diagnostics.length} loaded)</summary>
                <div style={{ marginTop: '4px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                  {imageGenMeta.lora_diagnostics.map((d: any, i: number) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '3px 6px', borderRadius: '6px', backgroundColor: d.loaded ? 'rgba(0,204,136,0.1)' : 'rgba(255,68,68,0.1)' }}>
                      <span style={{ color: d.loaded ? '#00cc88' : '#ff4444' }}>{d.loaded ? '✓' : '✗'}</span>
                      <span style={{ flex: 1, color: '#aaa' }}>{d.name}</span>
                      <span style={{ fontSize: '9px', fontFamily: 'monospace', color: '#666' }}>w={d.weight}</span>
                    </div>
                  ))}
                </div>
              </details>
            )}
            {imageGenMeta.injected_triggers?.length > 0 && (
              <details style={{ fontSize: '11px', color: '#888' }}>
                <summary style={{ cursor: 'pointer', color: '#666' }}>Injected trigger words ({imageGenMeta.injected_triggers.length})</summary>
                <div style={{ marginTop: '4px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                  {imageGenMeta.injected_triggers.map((t: any, i: number) => (
                    <div key={i} style={{ padding: '4px 8px', backgroundColor: '#0a0a0a', borderRadius: '6px', fontSize: '10px' }}>
                      <span style={{ color: '#b388ff', fontWeight: 500 }}>{t.lora}</span>
                      {t.method && <span style={{ color: '#555', marginLeft: '6px', fontSize: '9px' }}>({t.method})</span>}
                      <div style={{ color: '#aaa', marginTop: '2px', wordBreak: 'break-all' }}>{t.injected}</div>
                    </div>
                  ))}
                </div>
              </details>
            )}

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

            {/* Feedback learnings display */}
            <details style={{ fontSize: '11px', color: '#888' }}
              onToggle={(e: any) => { if (e.target.open) fetchFeedbackLearnings(); }}>
              <summary style={{ cursor: 'pointer', color: '#ff9900' }}>Feedback Learnings</summary>
              {feedbackLearnings ? (
                <div style={{ marginTop: '4px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {feedbackLearnings.negative_keywords?.length > 0 && (
                    <div>
                      <span style={{ fontSize: '9px', color: '#ff4444' }}>Avoiding:</span>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px', marginTop: '2px' }}>
                        {feedbackLearnings.negative_keywords.map((kw: string, i: number) => (
                          <span key={i} style={{ padding: '1px 6px', borderRadius: '8px', fontSize: '9px', backgroundColor: 'rgba(255,68,68,0.1)', color: '#ff4444', border: '1px solid rgba(255,68,68,0.2)' }}>{kw}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {feedbackLearnings.positive_keywords?.length > 0 && (
                    <div>
                      <span style={{ fontSize: '9px', color: '#00cc88' }}>Boosting:</span>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px', marginTop: '2px' }}>
                        {feedbackLearnings.positive_keywords.map((kw: string, i: number) => (
                          <span key={i} style={{ padding: '1px 6px', borderRadius: '8px', fontSize: '9px', backgroundColor: 'rgba(0,204,136,0.1)', color: '#00cc88', border: '1px solid rgba(0,204,136,0.2)' }}>{kw}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {(!feedbackLearnings.negative_keywords?.length && !feedbackLearnings.positive_keywords?.length) && (
                    <span style={{ color: '#444', fontSize: '10px' }}>No learnings yet — submit feedback to train the system</span>
                  )}
                  {(feedbackLearnings.negative_keywords?.length > 0 || feedbackLearnings.positive_keywords?.length > 0) && (
                    <button onClick={() => clearFeedbackLearnings('all')} style={{
                      marginTop: '4px', padding: '3px 8px', borderRadius: '6px', border: '1px solid #333',
                      backgroundColor: 'transparent', color: '#666', fontSize: '9px', cursor: 'pointer',
                    }}>Clear all learnings</button>
                  )}
                </div>
              ) : (
                <span style={{ color: '#444', fontSize: '10px' }}>Loading...</span>
              )}
            </details>
          </div>
        )}
      </div>
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
        {imageGenHistory.slice().reverse().map((h, i) => {
          const isExpanded = expandedHistoryIdx === i;
          return (
          <div key={i} style={{
            padding: '8px', borderRadius: '8px', backgroundColor: isExpanded ? '#1a1a2e' : '#111',
            border: `1px solid ${isExpanded ? 'rgba(179,136,255,0.3)' : '#222'}`, fontSize: '11px', cursor: 'pointer',
            transition: 'all 0.2s',
          }}
            onClick={() => setExpandedHistoryIdx(isExpanded ? null : i)}
          >
            {h.result_path && (
              <img
                src={`http://localhost:8000/file?path=${encodeURIComponent(h.result_path)}`}
                alt=""
                onClick={(e) => {
                  e.stopPropagation();
                  onOpenLightbox(`http://localhost:8000/file?path=${encodeURIComponent(h.result_path)}`);
                }}
                style={{ width: '100%', height: isExpanded ? '200px' : '120px', objectFit: 'cover', borderRadius: '6px', marginBottom: '6px', cursor: 'zoom-in', transition: 'height 0.2s' }}
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
            )}
            <div style={{ color: '#ccc', marginBottom: '4px', overflow: isExpanded ? 'visible' : 'hidden', textOverflow: isExpanded ? 'unset' : 'ellipsis', whiteSpace: isExpanded ? 'pre-wrap' : 'nowrap', fontSize: '11px', wordBreak: 'break-word' }}>
              {h.prompt?.replace(/^(score_9, score_8_up, score_7_up, |rating:\w+, )/, '').slice(0, isExpanded ? 500 : 80) || 'Unknown'}
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
            {/* Expanded details */}
            {isExpanded && (
              <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {h.settings && (
                  <div style={{ fontSize: '10px', color: '#666', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                    <span>steps: {h.settings.steps}</span>
                    <span>cfg: {h.settings.guidance_scale}</span>
                    <span>{h.settings.width}×{h.settings.height}</span>
                    {h.settings.loras?.length > 0 && <span>LoRAs: {h.settings.loras.join(', ')}</span>}
                  </div>
                )}
                {h.prompt_analysis?.estimated_focus && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px', marginTop: '2px' }}>
                    {h.prompt_analysis.estimated_focus.slice(0, 20).map((f: any, fi: number) => (
                      <span key={fi} style={{
                        padding: '1px 6px', borderRadius: '8px', fontSize: '9px',
                        backgroundColor: f.section !== 'shared' ? 'rgba(124,77,255,0.15)' : f.priority === 'high' ? 'rgba(0,204,136,0.15)' : 'rgba(80,80,80,0.15)',
                        color: f.section !== 'shared' ? '#b388ff' : f.priority === 'high' ? '#00cc88' : '#666',
                      }}>{f.text}</span>
                    ))}
                  </div>
                )}
                <div style={{ display: 'flex', gap: '4px', marginTop: '4px' }}>
                  <button onClick={(e) => {
                    e.stopPropagation();
                    let cleanPrompt = h.prompt?.replace(/^(score_9, score_8_up, score_7_up, |rating:\w+, )/, '') || '';
                    cleanPrompt = cleanPrompt
                      .replace(/,?\s*(?:on the (?:left|right)|in the (?:center|background))\s*/gi, ' ')
                      .replace(/^\d+ characters, (?:group shot, )?/i, '')
                      .replace(/^anime screencap, hand-drawn animation, cel shading, soft lighting, natural colour palette, expressive linework, high quality anime key visual, /i, '')
                      .replace(/^studio ghibli style, watercolour background, warm lighting, soft pastel tones, hand-painted, detailed environment, gentle linework, /i, '')
                      .replace(/^manga panel, black and white, screentone, detailed linework, dramatic shading, ink drawing, /i, '')
                      .replace(/^oil painting, visible brush strokes, impressionist lighting, textured canvas, rich earth tones, gallery quality, /i, '')
                      .replace(/^game concept art, splash art, dynamic composition, dramatic rim lighting, painterly rendering, detailed character design, /i, '')
                      .replace(/^photorealistic, RAW photo, 8k UHD, DSLR, professional lighting, studio quality, /i, '')
                      .replace(/^3d render, blender, unreal engine 5, octane render, volumetric lighting, subsurface scattering, PBR materials, video game character, detailed textures, sharp focus, /i, '')
                      .replace(/,\s*,+/g, ',')
                      .replace(/\s{2,}/g, ' ')
                      .trim().replace(/^,|,$/g, '').trim();
                    setImageGenPrompt(cleanPrompt);
                    if (h.settings?.seed) setImageGenSeed(h.settings.seed);
                    if (h.settings?.steps) setImageGenSteps(h.settings.steps);
                    if (h.settings?.guidance_scale) setImageGenCfg(h.settings.guidance_scale);
                  }} style={{ flex: 1, padding: '4px', borderRadius: '4px', border: '1px solid #333', backgroundColor: '#111', color: '#b388ff', cursor: 'pointer', fontSize: '9px' }}>
                    📋 Load Prompt
                  </button>
                  {h.result_path && (
                    <>
                      <button onClick={(e) => {
                        e.stopPropagation();
                        fetch(`http://localhost:8000/file/open?path=${encodeURIComponent(h.result_path)}`, { method: 'POST' });
                      }} style={{ flex: 1, padding: '4px', borderRadius: '4px', border: '1px solid #333', backgroundColor: '#111', color: '#888', cursor: 'pointer', fontSize: '9px' }}>
                        📂 Open
                      </button>
                      <button onClick={(e) => {
                        e.stopPropagation();
                        setUpscaleImagePath(h.result_path);
                        setImageGenSubTab('upscale');
                      }} style={{ flex: 1, padding: '4px', borderRadius: '4px', border: '1px solid #333', backgroundColor: '#111', color: '#888', cursor: 'pointer', fontSize: '9px' }}>
                        🔍 Upscale
                      </button>
                    </>
                  )}
                  <button onClick={async (e) => {
                    e.stopPropagation();
                    try {
                      await fetch(`http://localhost:8000/generate/history/${i}`, { method: 'DELETE' });
                      fetchImageHistory();
                      setExpandedHistoryIdx(null);
                    } catch {}
                  }} style={{ padding: '4px 6px', borderRadius: '4px', border: '1px solid rgba(255,68,68,0.3)', backgroundColor: 'rgba(255,68,68,0.05)', color: '#ff4444', cursor: 'pointer', fontSize: '9px' }}
                    title="Remove from history">
                    ✕
                  </button>
                </div>
              </div>
            )}
            {/* Collapsed action buttons */}
            {!isExpanded && h.result_path && (
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
                <button onClick={async (e) => {
                  e.stopPropagation();
                  try {
                    await fetch(`http://localhost:8000/generate/history/${i}`, { method: 'DELETE' });
                    fetchImageHistory();
                  } catch {}
                }} style={{ padding: '3px 6px', borderRadius: '4px', border: '1px solid rgba(255,68,68,0.2)', backgroundColor: 'transparent', color: '#ff4444', cursor: 'pointer', fontSize: '9px' }}
                  title="Remove from history">✕</button>
              </div>
            )}
          </div>
        );
        })}
      </div>
    </div>
  );
};
