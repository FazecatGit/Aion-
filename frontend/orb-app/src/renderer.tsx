import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import Orb from './Orb';
import './index.css';

type Message = { role: 'user' | 'ai'; text: string; isDiff?: boolean };
type OrbMode = 'idle' | 'querying' | 'agent-processing';
type ActiveMode = 'query' | 'agent';

function App() {
  const [mode, setMode] = useState<OrbMode>('idle');
  const [activeMode, setActiveMode] = useState<ActiveMode>('query');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [queryMode, setQueryMode] = useState<'auto'|'fast'|'deep'|'deep_semantic'|'both'>('auto');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [lastUserQuery, setLastUserQuery] = useState<string | null>(null);
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [pendingAgentEdit, setPendingAgentEdit] = useState<{ instruction: string; output: string; filePath: string } | null>(null);
  const computeSize = () => {
    const maxDim = Math.max(window.innerWidth, window.innerHeight);
    const raw = 2000 / maxDim;           // 2000 is arbitrary scale factor
    return Math.min(Math.max(raw, 200), 1200); // clamp between 200 and 1200px
  };
  const [baseSize, setBaseSize] = useState(computeSize());
  const [detailsContent, setDetailsContent] = useState<string | null>(null);
  const [ragChunkPrompt, setRagChunkPrompt] = useState<{ show: boolean; instruction: string; filePath: string } | null>(null);
  const [customChunkInput, setCustomChunkInput] = useState('');


  // update baseSize whenever window resizes so orb scales with window size
  useEffect(() => {
    const update = () => {
      setBaseSize(computeSize());
    };
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!input.trim()) return;

  const userMsg = input.trim();
  setInput('');
  setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
  
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
      
      setMessages(prev => [...prev, { role: 'ai', text: '[CODE AGENT] Processing your code request...' }]);
      const res = await fetch('http://localhost:8000/agent/edit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction: userMsg, file_path: selectedFilePath }),
      });
      const data = await res.json();
      
      if (data.status === 'pending_review') {
        // Show dry_run output and set pending edit; use resolved path returned by server
        const resolvedPath = data.file_path || selectedFilePath || '';
        setSelectedFilePath(resolvedPath);
        setMessages(prev => [...prev, { role: 'ai', text: `[DRY RUN PREVIEW]\n\n${data.dry_run_output}`, isDiff: true }]);
        setPendingAgentEdit({ instruction: userMsg, output: data.dry_run_output, filePath: resolvedPath });
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
        body: JSON.stringify({ question: userMsg, mode: queryMode }),
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

const handleRagChunkRetry = async (chunks: number) => {
  if (!ragChunkPrompt) return;
  
  setMode('agent-processing');
  setMessages(prev => [...prev, { role: 'ai', text: `[CODE AGENT] Retrying with ${chunks} RAG chunks...` }]);
  
  try {
    const res = await fetch('http://localhost:8000/agent/edit_with_chunks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        instruction: ragChunkPrompt.instruction, 
        file_path: ragChunkPrompt.filePath,
        max_chunks: chunks
      }),
    });
    const data = await res.json();
    
    if (data.status === 'pending_review') {
      const resolvedPath = data.file_path || ragChunkPrompt.filePath;
      setSelectedFilePath(resolvedPath);
      setMessages(prev => [...prev, { role: 'ai', text: `[DRY RUN PREVIEW]\n\n${data.dry_run_output}`, isDiff: true }]);
      setPendingAgentEdit({ instruction: ragChunkPrompt.instruction, output: data.dry_run_output, filePath: resolvedPath });
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
      {/* Top spacer (for nicer vertical balance) */}
      <div style={{ flex: 1 }} />

      {/* toolbar: upload + mode selector */}
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

        {activeMode === 'agent' && (
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flex: 1 }}>
            <input
              id="agent-file-picker"
              type="file"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = (e.target as HTMLInputElement).files?.[0];
                if (!f) return;
                // In Electron with sandbox:false, .path gives the full OS path
                const fullPath = (f as any).path || f.name;
                setSelectedFilePath(fullPath);
                (document.getElementById('agent-file-picker') as HTMLInputElement).value = '';
              }}
            />
            <button
              onClick={() => (document.getElementById('agent-file-picker') as HTMLInputElement)?.click()}
              style={{ padding: '6px 10px', borderRadius: '6px', backgroundColor: '#4422aa', color: '#fff', border: 'none', cursor: 'pointer', whiteSpace: 'nowrap' }}
              disabled={mode !== 'idle'}
            >
              📁 Select File
            </button>
            <input
              type="text"
              value={selectedFilePath || ''}
              onChange={e => setSelectedFilePath(e.target.value)}
              placeholder="Full path, e.g. C:\path\to\file.py"
              disabled={mode !== 'idle'}
              style={{ flex: 1, padding: '6px 8px', borderRadius: '6px', backgroundColor: '#111', color: '#fff', border: '1px solid #555', fontSize: '12px', minWidth: 0 }}
            />
          </div>
        )}
      </div>

      {/* Orb + Idle label (fixed center) */}
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

      {/* Chat history */}
        <div
        style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 30,
            width: '80%',
            maxWidth: '960px',
            maxHeight: '40vh',
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
        {messages.map((m, i) => (
            <div
            key={i}
            style={{
                padding: '12px 16px',
                backgroundColor: m.role === 'user' ? 'rgba(128, 0, 128, 0.5)' : 'rgba(255, 255, 255, 0.12)',  // Purple for user
                backdropFilter: 'blur(15px)',
                WebkitBackdropFilter: 'blur(15px)',
                borderRadius: '18px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: 'rgba(255, 255, 255, 0.95)',
                textShadow: '0 1px 2px rgba(0,0,0,0.5)', 
                maxWidth: '85%', 
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            }}
            >
            <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, wordBreak: 'break-word', fontFamily: m.isDiff ? 'monospace' : 'inherit' }}>
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
                        body: JSON.stringify({ question: lastUserQuery, prev_mode: queryMode })
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
              <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
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
                          confirmed: true
                        })
                      });
                      const result = await res.json();
                      setMessages(prev => [...prev, { role: 'ai', text: result.message || 'Changes applied successfully!' }]);
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
            )}
            </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

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
            <p style={{ marginBottom: '20px', lineHeight: '1.5' }}>
              LLM couldn't generate proper code changes. Would you like to retry with more RAG context?
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

      {/* Input bar fixed at bottom center */}
      <form
        onSubmit={handleSubmit}
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '80%',
          maxWidth: '640px',
          display: 'flex',
          gap: '10px',
          zIndex: 20,      // ensure input sits above orb
        }}
      >
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
        <button
          type="button"
          onClick={() => setActiveMode(activeMode === 'query' ? 'agent' : 'query')}
          disabled={mode !== 'idle'}
          style={{
            padding: '10px 14px',
            borderRadius: '12px',
            border: '2px solid',
            borderColor: activeMode === 'agent' ? '#ff00ff' : '#5533ff',
            backgroundColor: activeMode === 'agent' ? 'rgba(255,0,255,0.1)' : 'rgba(85,51,255,0.1)',
            color: activeMode === 'agent' ? '#ff00ff' : '#5533ff',
            cursor: 'pointer',
            fontSize: '12px',
            fontWeight: 'bold',
            textTransform: 'uppercase',
            transition: 'all 0.3s'
          }}
        >
          {activeMode === 'agent' ? '🤖 Agent' : '🔍 Query'}
        </button>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={activeMode === 'agent' ? 'Describe code changes...' : 'Ask Aion...'}
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
      </form>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
