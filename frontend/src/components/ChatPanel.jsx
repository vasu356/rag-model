/**
 * ChatPanel — Main chat interface for the RAGLab demo.
 *
 * Displays message history with source cards, quality metadata,
 * and an input area with retrieval-parameter controls.
 */

import { IconSend } from "./Icons";
import MdText from "./MdText";

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------
function ChatMessage({ msg, quality }) {
  const isLowConf =
    quality &&
    (quality.confidence || 0) < 0.4 &&
    msg.role === "ai";

  return (
    <div className={`msg msg-${msg.role}`}>
      <div className={`msg-avatar ${msg.role === "user" ? "user" : ""}`}>
        {msg.role === "ai" ? "AI" : msg.role === "user" ? "You" : "ℹ"}
      </div>
      <div className="msg-body">
        <MdText text={msg.content} />

        {/* Quality / confidence warning */}
        {isLowConf && (
          <div className="quality-note">
            ⚠ Low confidence answer ({Math.round(quality.confidence * 100)}%).
            Please verify with your document context.
          </div>
        )}

        {/* Quality metadata */}
        {quality && (
          <div className="response-meta">
            <div className="meta-row">
              <span className={`meta-pill ${isLowConf ? "warning" : ""}`}>
                Confidence: {Math.round(quality.confidence * 100)}%
              </span>
              {quality.grounding?.has_numbers && (
                <span className={`meta-pill ${(quality.grounding.grounded_ratio || 1) < 0.8 ? "warning" : ""}`}>
                  Numeric grounding: {Math.round((quality.grounding.grounded_ratio || 1) * 100)}%
                </span>
              )}
              <span className="meta-pill">
                Top-K: {quality.effective_top_k}
              </span>
            </div>
            {quality.query_rewrites?.length > 1 && (
              <div className="rewrite-row">
                <span className="meta-chip">
                  🔍 {quality.query_rewrites.length} query rewrites
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------
function TypingIndicator() {
  return (
    <div className="typing">
      <span />
      <span />
      <span />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function ChatPanel({
  messages,
  qualityMap,
  streaming,
  bottomRef,
  input,
  setInput,
  handleKey,
  query,
  topK,
  setTopK,
  alphaDense,
  setAlphaDense,
  alphaBm25,
  setAlphaBm25,
  answerLength,
  setAnswerLength,
}) {
  const lastMsg = messages[messages.length - 1];
  const showTyping = streaming && lastMsg?.content === "";

  return (
    <div className="chat-panel">
      {/* Messages area */}
      <div className="messages">
        {messages.map((m) => (
          <ChatMessage
            key={m.id}
            msg={m}
            quality={qualityMap?.[m.id]}
          />
        ))}
        {showTyping && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="input-row">
        <div className="input-wrap">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a question about your documents…"
            rows={1}
            disabled={streaming}
          />
          <button
            className="send-btn"
            onClick={query}
            disabled={!input.trim() || streaming}
          >
            <IconSend />
          </button>
        </div>

        {/* Parameter controls */}
        <div className="input-meta">
          <label>
            Top-K chunks
            <input
              type="range"
              min={1}
              max={8}
              value={topK}
              onChange={(e) => setTopK(+e.target.value)}
            />
            <strong>{topK}</strong>
          </label>

          <label>
            Semantic weight
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(alphaDense * 100)}
              onChange={(e) => {
                const d = +e.target.value / 100;
                setAlphaDense(d);
                setAlphaBm25(1 - d);
              }}
            />
            <strong>{Math.round(alphaDense * 100)}%</strong>
          </label>

          <label>
            Keyword weight
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(alphaBm25 * 100)}
              onChange={(e) => {
                const b = +e.target.value / 100;
                setAlphaBm25(b);
                setAlphaDense(1 - b);
              }}
            />
            <strong>{Math.round(alphaBm25 * 100)}%</strong>
          </label>

          <label>
            Answer length
            <select
              value={answerLength}
              onChange={(e) => setAnswerLength(e.target.value)}
            >
              <option value="short">Short</option>
              <option value="medium">Medium</option>
              <option value="detailed">Detailed</option>
            </select>
            <strong>
              {answerLength[0].toUpperCase() + answerLength.slice(1)}
            </strong>
          </label>

          <span className="muted controls-hint">
            ↵ send · ⇧↵ newline
          </span>
        </div>
      </div>
    </div>
  );
}