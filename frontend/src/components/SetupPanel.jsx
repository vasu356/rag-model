import { IconDoc, IconUpload, IconX } from "./Icons";

export default function SetupPanel({ phase, activeTab, setActiveTab, samples, selectedSamples, toggleSample, indexSamples, uploadFiles, setUploadFiles, fileRef, indexUpload }) {
  return (
    <div className="setup-panel">
      <div className="setup-hero"><div className="hero-orb" /><h1>Retrieval-Augmented Generation</h1><p>Upload documents or use samples. Ask questions. Watch the pipeline in action.</p></div>
      <div className="card setup-card">
        <div className="tabs">
          <button className={`tab ${activeTab === "samples" ? "active" : ""}`} onClick={() => setActiveTab("samples")}>Sample Docs</button>
          <button className={`tab ${activeTab === "upload" ? "active" : ""}`} onClick={() => setActiveTab("upload")}>Upload Files</button>
        </div>
        {activeTab === "samples" && (
          <div className="tab-content">
            <p className="hint">Select one or more pre-loaded documents to index:</p>
            <div className="sample-list">
              {samples.length === 0 && <p className="muted">Loading samples…</p>}
              {samples.map(s => (
                <label key={s.name} className={`sample-item ${selectedSamples.includes(s.name) ? "selected" : ""}`}>
                  <input type="checkbox" checked={selectedSamples.includes(s.name)} onChange={() => toggleSample(s.name)} />
                  <IconDoc />
                  <span className="sample-name">{s.name.replace(/_/g, " ").replace(/\.\w+$/, "")}</span>
                  <span className="sample-size">{(s.size / 1024).toFixed(1)} KB</span>
                </label>
              ))}
            </div>
            <button className="btn-primary" disabled={!selectedSamples.length || phase === "indexing"} onClick={indexSamples}>
              {phase === "indexing" ? <span className="spinner" /> : null}
              {phase === "indexing" ? "Indexing…" : "Index Selected Documents"}
            </button>
          </div>
        )}
        {activeTab === "upload" && (
          <div className="tab-content">
            <p className="hint">Upload TXT, MD, or PDF files:</p>
            <div className={`dropzone ${uploadFiles.length ? "has-files" : ""}`} onClick={() => fileRef.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); setUploadFiles(Array.from(e.dataTransfer.files)); }}>
              <input ref={fileRef} type="file" multiple accept=".txt,.md,.pdf" style={{ display: "none" }} onChange={e => setUploadFiles(Array.from(e.target.files))} />
              {uploadFiles.length === 0 ? (
                <><IconUpload /><p>Drop files here or click to browse</p><span className="muted">.txt · .md · .pdf</span></>
              ) : (
                <div className="file-list">
                  {uploadFiles.map((f, i) => (
                    <div key={i} className="file-chip"><IconDoc />{f.name}<button onClick={e => { e.stopPropagation(); setUploadFiles(prev => prev.filter((_, j) => j !== i)); }}><IconX /></button></div>
                  ))}
                </div>
              )}
            </div>
            <button className="btn-primary" disabled={!uploadFiles.length || phase === "indexing"} onClick={indexUpload}>
              {phase === "indexing" ? <span className="spinner" /> : <IconUpload />}
              {phase === "indexing" ? "Indexing…" : "Index Uploaded Files"}
            </button>
          </div>
        )}
      </div>
      <div className="pipeline">{["Chunk", "Embed", "Index", "Retrieve", "Generate"].map((step, i) => <div key={step} className="pipeline-step"><div className="pipeline-node">{i + 1}</div><span>{step}</span>{i < 4 && <div className="pipeline-arrow">→</div>}</div>)}</div>
    </div>
  );
}
