'use client';

import {
  ArrowLeft,
  ArrowRight,
  BookOpenText,
  Check,
  ChevronDown,
  Copy,
  GitCompareArrows,
  Search,
  Sparkles,
  X,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

type RecordSummary = {
  id: string;
  datasetIndex: number;
  bridgeIndex: number;
  problemId: string;
  topic?: string;
  difficulty?: number;
  questionPreview: string;
  bridgePreview: string;
  qualifyingBridgesForPrefix: number;
};

type IndexData = {
  definition: string;
  bridgeCount: number;
  prefixCount: number;
  replicatesPerCondition: number;
  records: RecordSummary[];
};

type Rollout = {
  replicate: number;
  correct: boolean;
  predictedAnswer?: string;
  targetAnswer?: string;
  tokenCount: number;
  finishReason?: string;
  text: string;
};

type PrefixData = {
  datasetIndex: number;
  problemId: string;
  topic?: string;
  difficulty?: number;
  question: string;
  responsePrefix: string;
  studentAnchor?: string;
  targetAnswer?: string;
  directCorrectCount: number;
  directRollouts: Rollout[];
};

type BridgeData = {
  datasetIndex: number;
  bridgeIndex: number;
  bridgeTokenCount: number;
  bridgeText: string;
  afterCorrectCount: number;
  afterRollouts: Rollout[];
};

function cleanTopic(topic?: string) {
  if (!topic) return 'Uncategorized';
  const parts = topic.split(' -> ');
  return parts.at(-1) ?? topic;
}

function averageTokenCount(items: Rollout[]) {
  return Math.round(items.reduce((sum, item) => sum + item.tokenCount, 0) / items.length);
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };
  return (
    <button className="icon-button" onClick={copy} title="Copy text" type="button">
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

function RolloutCard({ rollout, tone }: { rollout: Rollout; tone: 'bad' | 'good' }) {
  const [open, setOpen] = useState(false);
  return (
    <article className={`rollout-card ${tone} ${open ? 'open' : ''}`}>
      <button className="rollout-head" onClick={() => setOpen((value) => !value)} type="button">
        <span className="replicate-label">R{rollout.replicate + 1}</span>
        <span className="result-pill">{rollout.correct ? 'Correct' : 'Wrong'}</span>
        <span className="answer-preview" title={rollout.predictedAnswer}>
          {rollout.predictedAnswer || 'No parsed answer'}
        </span>
        <span className="token-count">{rollout.tokenCount.toLocaleString()} tok</span>
        <ChevronDown className="chevron" size={15} />
      </button>
      {open && (
        <div className="rollout-body">
          <div className="rollout-meta">
            <span>Target: {rollout.targetAnswer || '—'}</span>
            <span>Finish: {rollout.finishReason || '—'}</span>
            <CopyButton text={rollout.text} />
          </div>
          <pre>{rollout.text}</pre>
        </div>
      )}
    </article>
  );
}

function RolloutColumn({ title, subtitle, count, items, tone }: {
  title: string;
  subtitle: string;
  count: string;
  items: Rollout[];
  tone: 'bad' | 'good';
}) {
  return (
    <section className={`rollout-column ${tone}`}>
      <div className="column-title">
        <div>
          <p>{subtitle}</p>
          <h3>{title}</h3>
        </div>
        <div className="column-score">
          <strong>{count}</strong>
          <span>avg {averageTokenCount(items).toLocaleString()} tok</span>
        </div>
      </div>
      <div className="rollout-list">
        {items.map((rollout) => (
          <RolloutCard key={rollout.replicate} rollout={rollout} tone={tone} />
        ))}
      </div>
    </section>
  );
}

export default function Home() {
  const [index, setIndex] = useState<IndexData | null>(null);
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const [prefix, setPrefix] = useState<PrefixData | null>(null);
  const [bridge, setBridge] = useState<BridgeData | null>(null);
  const [showPrefix, setShowPrefix] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/data/index.json')
      .then((response) => response.json())
      .then((data: IndexData) => {
        setIndex(data);
        setSelectedId(data.records[0]?.id ?? '');
      });
  }, []);

  const filtered = useMemo(() => {
    if (!index) return [];
    const needle = query.trim().toLowerCase();
    if (!needle) return index.records;
    return index.records.filter((record) =>
      [record.id, record.problemId, record.topic, record.questionPreview, record.bridgePreview]
        .some((value) => value?.toLowerCase().includes(needle)),
    );
  }, [index, query]);

  const selected = filtered.find((record) => record.id === selectedId) ?? filtered[0];

  useEffect(() => {
    if (!selected) {
      setPrefix(null);
      setBridge(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    Promise.all([
      fetch(`/data/prefixes/${selected.datasetIndex}.json`).then((response) => response.json()),
      fetch(`/data/bridges/${selected.id}.json`).then((response) => response.json()),
    ]).then(([prefixData, bridgeData]: [PrefixData, BridgeData]) => {
      setPrefix(prefixData);
      setBridge(bridgeData);
      setLoading(false);
    });
  }, [selected?.id]);

  const move = useCallback((direction: number) => {
    if (!selected || filtered.length === 0) return;
    const current = filtered.findIndex((record) => record.id === selected.id);
    const next = (current + direction + filtered.length) % filtered.length;
    setSelectedId(filtered[next].id);
  }, [filtered, selected]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.tagName === 'INPUT') return;
      if (event.key === 'ArrowDown') move(1);
      if (event.key === 'ArrowUp') move(-1);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [move]);

  const selectedPosition = filtered.findIndex((record) => record.id === selected?.id);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><GitCompareArrows size={17} /></div>
        <div className="brand-copy">
          <strong>Bridge Lift Explorer</strong>
          <span>Qwen3-1.7B · H256 · strict subset</span>
        </div>
        <div className="top-stats">
          <div><strong>{index?.bridgeCount ?? '—'}</strong><span>bridges</span></div>
          <div><strong>{index?.prefixCount ?? '—'}</strong><span>questions</span></div>
          <div className="lift-stat"><strong>+100</strong><span>accuracy points</span></div>
        </div>
      </header>

      <div className="workspace">
        <aside className="record-panel">
          <div className="panel-intro">
            <p className="eyebrow"><Sparkles size={13} /> Complete reversals</p>
            <h1>From zero to perfect</h1>
            <p>Direct Student is 0/8. The selected bridge makes all 8 continuations correct.</p>
          </div>
          <label className="search-box">
            <Search size={15} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search question, topic, bridge…"
            />
            {query && <button onClick={() => setQuery('')} type="button"><X size={14} /></button>}
          </label>
          <div className="record-count"><span>{filtered.length} results</span><span>↑↓ to navigate</span></div>
          <nav className="record-list" aria-label="Qualifying bridges">
            {filtered.map((record) => (
              <button
                key={record.id}
                className={record.id === selected?.id ? 'record-item active' : 'record-item'}
                onClick={() => setSelectedId(record.id)}
                type="button"
              >
                <div className="record-kicker">
                  <span>#{record.datasetIndex} · B{record.bridgeIndex + 1}</span>
                  <span>{cleanTopic(record.topic)}</span>
                </div>
                <strong>{record.problemId}</strong>
                <p>{record.questionPreview}</p>
                <div className="mini-score"><span>0/8</span><i /><span>8/8</span></div>
              </button>
            ))}
            {filtered.length === 0 && <div className="empty-state">No bridge matches this search.</div>}
          </nav>
        </aside>

        <section className="detail-panel">
          {!prefix || !bridge || loading ? (
            <div className="loading-state"><span /><p>Loading rollout evidence…</p></div>
          ) : (
            <>
              <div className="detail-nav">
                <button onClick={() => move(-1)} type="button"><ArrowLeft size={15} /> Previous</button>
                <span>{selectedPosition + 1} / {filtered.length}</span>
                <button onClick={() => move(1)} type="button">Next <ArrowRight size={15} /></button>
              </div>

              <article className="question-card">
                <div className="question-meta">
                  <span>Dataset #{prefix.datasetIndex}</span>
                  <span>{prefix.problemId}</span>
                  {prefix.difficulty != null && <span>Difficulty {prefix.difficulty}</span>}
                  <span>{cleanTopic(prefix.topic)}</span>
                </div>
                <div className="section-label"><BookOpenText size={14} /> Problem</div>
                <p className="question-text">{prefix.question}</p>
                <button className="prefix-toggle" onClick={() => setShowPrefix((value) => !value)} type="button">
                  {showPrefix ? 'Hide' : 'Show'} shared response prefix
                  <ChevronDown className={showPrefix ? 'rotate' : ''} size={15} />
                </button>
                {showPrefix && <pre className="prefix-text">{prefix.responsePrefix}</pre>}
              </article>

              <article className="bridge-card">
                <div className="bridge-heading">
                  <div>
                    <div className="section-label"><Sparkles size={14} /> Intervention</div>
                    <h2>Bridge {bridge.bridgeIndex + 1}</h2>
                  </div>
                  <div className="bridge-badges">
                    <span>{bridge.bridgeTokenCount} tokens</span>
                    <span>{selected?.qualifyingBridgesForPrefix}/8 perfect bridges for this question</span>
                    <CopyButton text={bridge.bridgeText} />
                  </div>
                </div>
                <pre>{bridge.bridgeText}</pre>
              </article>

              <div className="comparison-banner">
                <div><span>Direct</span><strong>0%</strong></div>
                <div className="lift-line"><i /><span>+100 pp</span><i /></div>
                <div><span>After bridge</span><strong>100%</strong></div>
              </div>

              <div className="rollout-grid">
                <RolloutColumn
                  title="Original rollouts"
                  subtitle="Student from prefix"
                  count="0 / 8"
                  items={prefix.directRollouts}
                  tone="bad"
                />
                <RolloutColumn
                  title="Bridge-conditioned rollouts"
                  subtitle="Student from prefix + bridge"
                  count="8 / 8"
                  items={bridge.afterRollouts}
                  tone="good"
                />
              </div>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
