import { useEffect, useRef, useState } from 'react';
import { createSimulatedClient, normalizeError } from './api/client';
import type { InferenceError } from './api/client';
import type { AdaptiveResponse, Category } from './api/types';
import type { Scenario } from './mocks/fixtures';
import { Header } from './components/Header';
import { Settings } from './components/Settings';
import { AdvancedControls, validGeneration } from './components/AdvancedControls';
import { Composer } from './components/Composer';
import { ResponsePanel } from './components/ResponsePanel';
import { RequestError } from './components/RequestError';
import { useTheme } from './useTheme';

interface Turn { id: number; prompt: string; category: Category; manual: boolean; response?: AdaptiveResponse; error?: InferenceError }
const examples = ['How does adaptive routing work?', 'When does routing use a fallback?', 'Explain validation and escalation.'];

export default function App() {
  const [page, setPage] = useState<'chat' | 'activity'>('chat');
  const [sidebar, setSidebar] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [theme, setTheme, resolvedTheme] = useTheme();
  const [manual, setManual] = useState(false);
  const [category, setCategory] = useState<Category>('qa');
  const [tokens, setTokens] = useState('256');
  const [temperature, setTemperature] = useState('1.0');
  const [scenario, setScenario] = useState<Scenario>('normal');
  const [draft, setDraft] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState(false);
  const busy = useRef(false);
  const nextId = useRef(0);
  const composer = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (turns.length && page === 'chat') end.current?.scrollIntoView?.({ block: 'end', behavior: 'instant' });
  }, [turns, page]);

  function newChat() {
    if (busy.current) return;
    setTurns([]); setDraft(''); setPage('chat'); setSidebar(false);
    requestAnimationFrame(() => composer.current?.focus());
  }

  async function send() {
    if (busy.current || !draft.trim() || draft.length > 12_000 || !validGeneration(tokens, temperature)) return;
    busy.current = true;
    setPending(true);
    const id = ++nextId.current;
    // Auto-detect is a presentation fixture (QA), not a classifier or backend feature.
    const effectiveCategory = manual ? category : 'qa';
    const turn: Turn = { id, prompt: draft, category: effectiveCategory, manual };
    setTurns(previous => [...previous, turn]);
    setDraft('');
    try {
      const response = await createSimulatedClient(scenario).infer({ prompt: draft, category: effectiveCategory, max_output_tokens: Number(tokens), temperature: Number(temperature), quality_threshold: 0.80 });
      setTurns(previous => previous.map(item => item.id === id ? { ...item, response } : item));
    } catch (failure) {
      setTurns(previous => previous.map(item => item.id === id ? { ...item, error: normalizeError(failure) } : item));
    } finally { busy.current = false; setPending(false); }
  }

  return <div className="app" data-theme={resolvedTheme} data-theme-mode={theme}>
    <a className="skip-link" href="#main">Skip to conversation</a>
    <aside className={`sidebar ${sidebar ? 'sidebar-open' : ''}`} aria-label="Conversations">
      <a className="brand" href="#main" onClick={() => { setPage('chat'); setSidebar(false); }}><span className="mark">R↗</span><span><strong>RoutLLM</strong><small>Adaptive LLM Gateway</small></span></a>
      <button className="new-chat" onClick={newChat} disabled={pending}><span aria-hidden="true">＋</span> New chat</button>
      <nav aria-label="Workspace"><button aria-current={page === 'chat' ? 'page' : undefined} onClick={() => { setPage('chat'); setSidebar(false); }}>Conversation <span aria-hidden="true">↗</span></button><button aria-current={page === 'activity' ? 'page' : undefined} onClick={() => { setPage('activity'); setSidebar(false); }}>Activity <span className="small-label">Soon</span></button></nav>
      <div className="recent"><h2>Recent <span>Demo examples</span></h2>{examples.map((example, index) => <button key={example} disabled={pending} onClick={() => { setDraft(example); setScenario((['normal', 'fallback', 'escalation'] as const)[index]); setPage('chat'); setSidebar(false); requestAnimationFrame(() => composer.current?.focus()); }}>{example}</button>)}<p>Example prompts, not saved history.<br />Your conversation stays in memory.</p></div>
      <div className="sidebar-footer"><span className="small-label">ROUTING, MADE VISIBLE</span><p>One conversation.<br />The right route behind it.</p></div>
    </aside>
    <div className="workspace"><Header onMenu={() => setSidebar(value => !value)} sidebarOpen={sidebar}><Settings advanced={advanced} onAdvanced={setAdvanced} theme={theme} onTheme={setTheme} scenario={scenario} onScenario={setScenario} pending={pending} /></Header>
      <div className={`workspace-body ${advanced ? 'with-controls' : ''}`}>
        <main id="main" className="main" tabIndex={-1}>
          {page === 'activity' ? <section className="activity"><span className="small-label">ACTIVITY / INTEGRATION PENDING</span><h1>The story behind every request.</h1><p>Durable telemetry integration will appear here.</p><p>PostgreSQL-backed activity is planned for a later phase. No request counts, costs, or reliability metrics are loaded or fabricated.</p><button className="text-button" onClick={() => setPage('chat')}>Back to conversation →</button></section> : <>
            <div className="conversation" aria-label="Conversation">
              {turns.length === 0 ? <section className="welcome"><span className="welcome-mark" aria-hidden="true">R↗</span><p className="small-label">INTELLIGENCE, ROUTED.</p><h1>A conversation.<br /><span>A smarter path.</span></h1><p>Ask a question. Explore the route behind the answer.</p><div className="suggestions">{examples.slice(0, 2).map(example => <button key={example} onClick={() => { setDraft(example); composer.current?.focus(); }}>{example}<span aria-hidden="true">↗</span></button>)}</div></section> : turns.map(turn => <section className="turn" key={turn.id} aria-label={`Message ${turn.id}`}>
                <div className="user-message"><span className="message-author">You</span><p>{turn.prompt}</p></div>
                {turn.response ? <ResponsePanel response={turn.response} category={turn.category} manual={turn.manual} /> : turn.error ? <RequestError error={turn.error} /> : <div className="assistant-pending"><span className="mark">R↗</span><span>Preparing response…</span></div>}
              </section>)}
              <div ref={end} className="conversation-end" />
            </div>
            <p className="sr-only" role="status">{pending ? 'Preparing response. No provider calls.' : turns.at(-1)?.error ? 'Request failed. No automatic retry.' : turns.length ? 'Response ready.' : ''}</p>
            <Composer ref={composer} value={draft} onChange={setDraft} onSend={send} pending={pending} validSettings={validGeneration(tokens, temperature)} onOpenControls={() => setAdvanced(true)} />
          </>}
        </main>
        {advanced && <AdvancedControls onClose={() => setAdvanced(false)} manual={manual} onManual={setManual} category={category} onCategory={setCategory} tokens={tokens} onTokens={setTokens} temperature={temperature} onTemperature={setTemperature} pending={pending} />}
      </div>
    </div>
  </div>;
}
