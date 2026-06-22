import { IconDoc } from "./Icons";

export default function SidebarPanel({ sessionMeta, topK, alphaDense, alphaBm25, answerLength }) {
  return (
    <aside className="sidebar">
      <div className="card sidebar-card">
        <h3>Session</h3>
        <div className="stat"><span>Documents</span><strong>{sessionMeta?.doc_count}</strong></div>
        <div className="stat"><span>Chunks</span><strong>{sessionMeta?.chunk_count}</strong></div>
        <div className="stat"><span>Top-K</span><strong>{topK}</strong></div>
        <div className="stat"><span>Dense weight</span><strong>{Math.round(alphaDense * 100)}%</strong></div>
        <div className="stat"><span>Keyword weight</span><strong>{Math.round(alphaBm25 * 100)}%</strong></div>
        <div className="stat"><span>Answer length</span><strong>{answerLength[0].toUpperCase() + answerLength.slice(1)}</strong></div>
        <div className="divider" />
        <h3>Indexed files</h3>
        {sessionMeta?.doc_names?.map(n => (
          <div key={n} className="indexed-file">
            <IconDoc />
            <span>{n.replace(/_/g, " ").replace(/\.\w+$/, "")}</span>
          </div>
        ))}
        <div className="divider" />
        <h3>Stack</h3>
        <div className="stack-list">
          {[
            "LlamaIndex",
            "FastAPI",
            "llama-3.1-8b-instant",
            "bge-small-en-v1.5",
            "Cross-encoder reranking",
            "SSE streaming",
          ].map(t => <span key={t} className="tag">{t}</span>)}
        </div>
      </div>
    </aside>
  );
}
