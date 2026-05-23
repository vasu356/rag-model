import { IconDoc } from "./Icons";

export default function SidebarPanel({ sessionMeta, topK, alphaDense, alphaBm25, answerLength }) {
  return (
    <aside className="sidebar">
      <div className="card sidebar-card">
        <h3>Session</h3>
        <div className="stat"><span>Documents</span><strong>{sessionMeta?.doc_count}</strong></div>
        <div className="stat"><span>Chunks</span><strong>{sessionMeta?.chunk_count}</strong></div>
        <div className="stat"><span>Top-K</span><strong>{topK}</strong></div>
        <div className="stat"><span>Dense Weight</span><strong>{Math.round(alphaDense * 100)}%</strong></div>
        <div className="stat"><span>Keyword Weight</span><strong>{Math.round(alphaBm25 * 100)}%</strong></div>
        <div className="stat"><span>Answer Length</span><strong>{answerLength[0].toUpperCase() + answerLength.slice(1)}</strong></div>
        <div className="divider" />
        <h3>Indexed Files</h3>
        {sessionMeta?.doc_names?.map(n => <div key={n} className="indexed-file"><IconDoc /><span>{n.replace(/_/g, " ").replace(/\.\w+$/, "")}</span></div>)}
        <div className="divider" />
        <h3>Stack</h3>
        <div className="stack-list">{["LlamaIndex", "FastAPI", "llama3-8b-8192", "bge-small-en-v1.5", "SSE Streaming"].map(t => <span key={t} className="tag">{t}</span>)}</div>
      </div>
    </aside>
  );
}
