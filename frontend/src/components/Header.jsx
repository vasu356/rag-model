import { IconChip } from "./Icons";

export default function Header({ sessionMeta, onReset }) {
  return (
    <header>
      <div className="header-inner">
        <div className="logo">
          <div className="logo-chip">
            <IconChip />
          </div>
          <div>
            <span className="logo-title">RAG<span className="accent">Lab</span></span>
            <span className="logo-sub">LlamaIndex · Groq · FastAPI</span>
          </div>
        </div>
        <div className="header-right">
          {sessionMeta && (
            <div className="session-badge">
              <span className="dot" />
              {sessionMeta.doc_count} doc{sessionMeta.doc_count !== 1 ? "s" : ""} · {sessionMeta.chunk_count} chunks
            </div>
          )}
          <a href="https://github.com/vasu356/rag-model" className="gh-link" target="_blank" rel="noreferrer">
            GitHub
          </a>
          {!!sessionMeta && (
            <button type="button" className="btn-ghost" onClick={onReset}>
              Start new session
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
