import { useState, useRef, useEffect, useCallback } from "react";
import Header from "./components/Header";
import SetupPanel from "./components/SetupPanel";
import ChatPanel from "./components/ChatPanel";
import SidebarPanel from "./components/SidebarPanel";

const API = "http://localhost:8000";
const ROLE = { USER: "user", AI: "ai", SYSTEM: "system" };

export default function App() {
  const [phase, setPhase] = useState("idle");
  const [sessionId, setSessionId] = useState(null);
  const [sessionMeta, setSessionMeta] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSamples, setSelectedSamples] = useState([]);
  const [uploadFiles, setUploadFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [qualityMap, setQualityMap] = useState({});
  const [activeTab, setActiveTab] = useState("samples");
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

  useEffect(() => {
    fetch(`${API}/samples`).then(r => r.json()).then(d => setSamples(d.samples || [])).catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  const appendMessage = useCallback((role, content, id) => {
    const msgId = id || nextMsgId();
    setMessages(prev => [...prev, { role, content, id: msgId }]);
    return msgId;
  }, [nextMsgId]);

  const appendToken = useCallback((msgId, token) => {
    setMessages(prev => prev.map(m => (m.id === msgId ? { ...m, content: m.content + token } : m)));
  }, []);

  const indexSamples = async () => {
    if (!selectedSamples.length) return;
    setPhase("indexing");
    try {
      const res = await fetch(`${API}/index/samples`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(selectedSamples) });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSessionId(data.session_id);
      setSessionMeta(data);
      setPhase("ready");
      setMessages([{ role: ROLE.SYSTEM, content: `✓ Indexed **${data.doc_count}** document${data.doc_count !== 1 ? "s" : ""} → **${data.chunk_count}** chunks. Ask me anything about the content.`, id: nextMsgId() }]);
    } catch (e) {
      setPhase("idle");
      alert("Indexing failed: " + e.message);
    }
  };

  const indexUpload = async () => {
    if (!uploadFiles.length) return;
    setPhase("indexing");
    const form = new FormData();
    uploadFiles.forEach(f => form.append("files", f));
    try {
      const res = await fetch(`${API}/index/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSessionId(data.session_id);
      setSessionMeta(data);
      setPhase("ready");
      setMessages([{ role: ROLE.SYSTEM, content: `✓ Uploaded & indexed **${data.doc_count}** file${data.doc_count !== 1 ? "s" : ""} → **${data.chunk_count}** chunks. Ask me anything.`, id: nextMsgId() }]);
    } catch (e) {
      setPhase("idle");
      alert("Upload failed: " + e.message);
    }
  };

  const query = async () => {
    const q = input.trim();
    if (!q || !sessionId || streaming) return;
    setInput("");
    setStreaming(true);
    appendMessage(ROLE.USER, q);
    const aiId = nextMsgId();
    setMessages(prev => [...prev, { role: ROLE.AI, content: "", id: aiId }]);

    try {
      const chatHistory = [
        ...messages
          .filter(m => m.role === ROLE.USER || m.role === ROLE.AI)
          .slice(-12)
          .map(m => ({ role: m.role, content: m.content })),
        { role: ROLE.USER, content: q },
      ];

      const res = await fetch(`${API}/query/stream`, {
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
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const lines = decoder.decode(value).split("\n").filter(l => l.startsWith("data: "));
        for (const line of lines) {
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === "quality") {
              setQualityMap(prev => ({ ...prev, [aiId]: evt.data }));
            } else if (evt.type === "token") {
              appendToken(aiId, evt.data);
            }
          } catch {}
        }
      }
    } catch (e) {
      appendToken(aiId, `\n\n⚠ Error: ${e.message}`);
    } finally {
      setStreaming(false);
    }
  };

  const handleKey = e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); query(); }
  };

  const reset = async () => {
    if (sessionId) await fetch(`${API}/session/${sessionId}`, { method: "DELETE" }).catch(() => {});
    setPhase("idle");
    setSessionId(null); setSessionMeta(null); setMessages([]); setSelectedSamples([]); setUploadFiles([]); setQualityMap({}); msgSeqRef.current = 0;
  };

  const toggleSample = name => setSelectedSamples(prev => (prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]));

  return (
    <div className="app">
      <Header sessionMeta={sessionMeta} onReset={reset} />
      <main>
        {phase !== "ready" && <SetupPanel phase={phase} activeTab={activeTab} setActiveTab={setActiveTab} samples={samples} selectedSamples={selectedSamples} toggleSample={toggleSample} indexSamples={indexSamples} uploadFiles={uploadFiles} setUploadFiles={setUploadFiles} fileRef={fileRef} indexUpload={indexUpload} />}
        {phase === "ready" && (
          <div className="chat-layout">
            <ChatPanel messages={messages} qualityMap={qualityMap} streaming={streaming} bottomRef={bottomRef} input={input} setInput={setInput} handleKey={handleKey} query={query} topK={topK} setTopK={setTopK} alphaDense={alphaDense} setAlphaDense={setAlphaDense} alphaBm25={alphaBm25} setAlphaBm25={setAlphaBm25} answerLength={answerLength} setAnswerLength={setAnswerLength} />
            <SidebarPanel sessionMeta={sessionMeta} topK={topK} alphaDense={alphaDense} alphaBm25={alphaBm25} answerLength={answerLength} />
          </div>
        )}
      </main>
    </div>
  );
}
