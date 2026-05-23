import { IconSend } from "./Icons";
import MdText from "./MdText";

export default function ChatPanel({ messages, qualityMap, streaming, bottomRef, input, setInput, handleKey, query, topK, setTopK, alphaDense, setAlphaDense, alphaBm25, setAlphaBm25, answerLength, setAnswerLength }) {
  return (
    <div className="chat-panel">
      <div className="messages">
        {messages.map(m => (
          <div key={m.id} className={`msg msg-${m.role}`}>
            {m.role === "ai" && <div className="msg-avatar">AI</div>}
            {m.role === "user" && <div className="msg-avatar user">You</div>}
            <div className="msg-body">
              <MdText text={m.content} />
              {m.role === "ai" && qualityMap?.[m.id] && (qualityMap[m.id].confidence || 0) < 0.4 && (
                <div className="quality-note">Low confidence answer. Please verify with your document context.</div>
              )}
            </div>
          </div>
        ))}
        {streaming && messages[messages.length - 1]?.content === "" && <div className="typing"><span /><span /><span /></div>}
        <div ref={bottomRef} />
      </div>
      <div className="input-row">
        <div className="input-wrap">
          <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKey} placeholder="Ask a question about your documents…" rows={1} disabled={streaming} />
          <button className="send-btn" onClick={query} disabled={!input.trim() || streaming}><IconSend /></button>
        </div>
        <div className="input-meta">
          <label>Top-K chunks:<input type="range" min={1} max={8} value={topK} onChange={e => setTopK(+e.target.value)} /><strong>{topK}</strong></label>
          <label>Dense weight:<input type="range" min={0} max={100} value={Math.round(alphaDense * 100)} onChange={e => { const d = +e.target.value / 100; setAlphaDense(d); setAlphaBm25(1 - d); }} /><strong>{Math.round(alphaDense * 100)}%</strong></label>
          <label>Keyword weight:<input type="range" min={0} max={100} value={Math.round(alphaBm25 * 100)} onChange={e => { const b = +e.target.value / 100; setAlphaBm25(b); setAlphaDense(1 - b); }} /><strong>{Math.round(alphaBm25 * 100)}%</strong></label>
          <label>Answer length:<select value={answerLength} onChange={e => setAnswerLength(e.target.value)}><option value="short">Short</option><option value="medium">Medium</option><option value="detailed">Detailed</option></select><strong>{answerLength[0].toUpperCase() + answerLength.slice(1)}</strong></label>
          <span className="muted controls-hint">↵ to send · ⇧↵ newline</span>
        </div>
      </div>
    </div>
  );
}
