/**
 * RAGLab — Main Application Component
 *
 * State management for the full RAG pipeline:
 *   Setup → Index → Query (SSE streaming) → Display
 */

import { useState, useRef, useEffect, useCallback } from "react";
import Header from "./components/Header";
import SetupPanel from "./components/SetupPanel";
import ChatPanel from "./components/ChatPanel";
import SidebarPanel from "./components/SidebarPanel";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const ROLE = { USER: "user", AI: "ai", SYSTEM: "system" };

/**
 * Safely parse a JSON event from an SSE data line.
 * Returns null on failure.
 */
function parseSSE(line) {
  if (!line.startsWith("data: ")) return null;
  try {
    return JSON.parse(line.slice(6));
  } catch {
    return null;
  }
}

export default function App() {
  // ---- State -------------------------------------------------------------
  const [phase, setPhase] = useState("idle"); // idle | indexing | ready
  const [sessionId, setSessionId] = useState(null);
  const [sessionMeta, setSessionMeta] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSamples, setSelectedSamples] = useState([]);
  const [uploadFiles, setUploadFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [qualityMap, setQualityMap] = useState({});
  const [sourceMap, setSourceMap] = useState({});
  const [activeTab, setActiveTab] = useState("samples");
  const [error, setError] = useState(null);

  // Retrieval parameters
  const [topK, setTopK] = useState(4);
  const [alphaDense, setAlphaDense] = useState(0.65);
  const [alphaBm25, setAlphaBm25] = useState(0.35);
  const [answerLength, setAnswerLength] = useState("medium");

  const bottomRef = useRef(null);
  const fileRef = useRef(null);
  const msgSeqRef = useRef(0);

  const nextMsgId = useCallback(() => {
    msgSeqRef.current += 1;
    return `m_${Date.now()}_${msgSeqRef.current}`;
  }, []);

  // ---- Fetch samples on mount -------------------------------------------
  useEffect(() => {
    fetch(`${API_BASE}/samples`)
      .then((r) => r.json())
      .then((d) => setSamples(d.samples || []))
      .catch(() => {});
  }, []);

  // ---- Auto-scroll -------------------------------------------------------
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  // ---- Message helpers ---------------------------------------------------
  const appendMessage = useCallback(
    (role, content, id) => {
      const msgId = id || nextMsgId();
      setMessages((prev) => [...prev, { role, content, id: msgId }]);
      return msgId;
    },
    [nextMsgId],
  );

  const appendToken = useCallback((msgId, token) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === msgId ? { ...m, content: m.content + token } : m,
      ),
    );
  }, []);

  // ---- Indexing ----------------------------------------------------------
  const indexSamples = async () => {
    if (!selectedSamples.length) return;
    setPhase("indexing");
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/index/samples`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(selectedSamples),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSessionId(data.session_id);
      setSessionMeta(data);
      setPhase("ready");
      setMessages([
        {
          role: ROLE.SYSTEM,
          content: `✓ Indexed **${data.doc_count}** document${data.doc_count !== 1 ? "s" : ""} → **${data.chunk_count}** chunks. Ask me anything about the content.`,
          id: nextMsgId(),
        },
      ]);
    } catch (e) {
      setPhase("idle");
      setError(`Indexing failed: ${e.message}`);
    }
  };

  const indexUpload = async () => {
    if (!uploadFiles.length) return;
    setPhase("indexing");
    setError(null);
    const form = new FormData();
    uploadFiles.forEach((f) => form.append("files", f));
    try {
      const res = await fetch(`${API_BASE}/index/upload`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSessionId(data.session_id);
      setSessionMeta(data);
      setPhase("ready");
      setMessages([
        {
          role: ROLE.SYSTEM,
          content: `✓ Uploaded & indexed **${data.doc_count}** file${data.doc_count !== 1 ? "s" : ""} → **${data.chunk_count}** chunks. Ask me anything.`,
          id: nextMsgId(),
        },
      ]);
    } catch (e) {
      setPhase("idle");
      setError(`Upload failed: ${e.message}`);
    }
  };

  // ---- Query / SSE streaming --------------------------------------------
  const query = async () => {
    const q = input.trim();
    if (!q || !sessionId || streaming) return;

    setInput("");
    setStreaming(true);
    setError(null);

    // Add user message immediately
    appendMessage(ROLE.USER, q);
    const aiId = nextMsgId();
    setMessages((prev) => [...prev, { role: ROLE.AI, content: "", id: aiId }]);

    try {
      // Build chat history for context (last 12 turns)
      const chatHistory = [
        ...messages
          .filter((m) => m.role === ROLE.USER || m.role === ROLE.AI)
          .slice(-12)
          .map((m) => ({ role: m.role, content: m.content })),
        { role: ROLE.USER, content: q },
      ];

      const res = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          question: q,
          top_k: topK,
          alpha_dense: alphaDense,
          alpha_bm25: alphaBm25,
          answer_length: answerLength,
          chat_history: chatHistory,
        }),
      });

      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(errBody || `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk
          .split("\n")
          .filter((l) => l.startsWith("data: "));

        for (const line of lines) {
          const evt = parseSSE(line);
          if (!evt) continue;

          switch (evt.type) {
            case "sources":
              setSourceMap((prev) => ({ ...prev, [aiId]: evt.data }));
              break;
            case "quality":
              setQualityMap((prev) => ({ ...prev, [aiId]: evt.data }));
              break;
            case "token":
              appendToken(aiId, evt.data);
              break;
            case "error":
              appendToken(aiId, `\n\n⚠ Error during generation: ${evt.data}`);
              break;
            default:
              break;
          }
        }
      }
    } catch (e) {
      appendToken(aiId, `\n\n⚠ Error: ${e.message}`);
    } finally {
      setStreaming(false);
    }
  };

  // ---- Event handlers ----------------------------------------------------
  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      query();
    }
  };

  const reset = async () => {
    if (sessionId) {
      try {
        await fetch(`${API_BASE}/session/${sessionId}`, { method: "DELETE" });
      } catch {
        // best-effort cleanup
      }
    }
    setPhase("idle");
    setSessionId(null);
    setSessionMeta(null);
    setMessages([]);
    setSelectedSamples([]);
    setUploadFiles([]);
    setQualityMap({});
    setSourceMap({});
    setError(null);
    msgSeqRef.current = 0;
  };

  const toggleSample = (name) =>
    setSelectedSamples((prev) =>
      prev.includes(name)
        ? prev.filter((n) => n !== name)
        : [...prev, name],
    );

  // ---- Render ------------------------------------------------------------
  return (
    <div className="app">
      <Header sessionMeta={sessionMeta} onReset={reset} />

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <main>
        {phase !== "ready" && (
          <SetupPanel
            phase={phase}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            samples={samples}
            selectedSamples={selectedSamples}
            toggleSample={toggleSample}
            indexSamples={indexSamples}
            uploadFiles={uploadFiles}
            setUploadFiles={setUploadFiles}
            fileRef={fileRef}
            indexUpload={indexUpload}
          />
        )}

        {phase === "ready" && (
          <div className="chat-layout">
            <ChatPanel
              messages={messages}
              qualityMap={qualityMap}
              streaming={streaming}
              bottomRef={bottomRef}
              input={input}
              setInput={setInput}
              handleKey={handleKey}
              query={query}
              topK={topK}
              setTopK={setTopK}
              alphaDense={alphaDense}
              setAlphaDense={setAlphaDense}
              alphaBm25={alphaBm25}
              setAlphaBm25={setAlphaBm25}
              answerLength={answerLength}
              setAnswerLength={setAnswerLength}
            />
            <SidebarPanel
              sessionMeta={sessionMeta}
              topK={topK}
              alphaDense={alphaDense}
              alphaBm25={alphaBm25}
              answerLength={answerLength}
            />
          </div>
        )}
      </main>
    </div>
  );
}