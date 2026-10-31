import { useEffect, useMemo, useState } from "react";
import { analyze, fetchImage } from "./api";

const LABEL = {
  Exposure2012: ["Exposure", 5], Contrast2012: ["Contrast", 100], Highlights2012: ["Highlights", 100],
  Shadows2012: ["Shadows", 100], Whites2012: ["Whites", 100], Blacks2012: ["Blacks", 100],
  Saturation: ["Saturation", 100], Vibrance: ["Vibrance", 100], Clarity2012: ["Clarity", 100],
  Texture: ["Texture", 100], Dehaze: ["Dehaze", 100], Tint: ["Tint", 150],
  IncrementalTemperature: ["Warmth", 100], IncrementalTint: ["Tint shift", 100],
  PostCropVignetteAmount: ["Vignette", 100], GrainAmount: ["Grain", 100], Sharpness: ["Sharpening", 150],
};
const STEPS = [["metadata", "In the file", "Settings saved by the editing app"],
  ["fitted", "Original", "Matched and compared"], ["estimate", "Estimate", "Best guess"]];
const HEAD = { metadata: "Exact settings found", fitted: "Matched to the original", estimate: "Rough guess" };

const sign = (n) => (n > 0 ? "+" : "");
const fmt = (k, v) => { const n = Number(v); if (Number.isNaN(n)) return String(v);
  if (k === "Temperature") return `${Math.round(n)} K`;
  return k === "Exposure2012" ? `${sign(n)}${n.toFixed(2)}` : `${sign(n)}${Math.round(n * 100) / 100}`; };
const word = (a) => (Math.abs(a) < 25 ? "slightly" : Math.abs(a) < 60 ? "noticeably" : "strongly");

function sentences(p, u = {}) {
  const g = (k) => (p[k] === undefined ? 0 : Number(p[k])), out = [];
  const e = g("Exposure2012"), c = g("Contrast2012"), s = g("Saturation"), w = g("IncrementalTemperature");
  if (Math.abs(e) >= 0.15) out.push(`${e > 0 ? "Brighter" : "Darker"} by about ${Math.abs(e).toFixed(1)} stop${Math.abs(e) >= 1.5 ? "s" : ""}`);
  if (Math.abs(c) >= 10) out.push(`Contrast ${word(c)} ${c > 0 ? "increased" : "reduced"}`);
  if (s <= -90) out.push("Converted to black and white");
  else if (Math.abs(s) >= 10) out.push(`Colors ${word(s)} ${s > 0 ? "more vivid" : "more muted"}`);
  if (Math.abs(w) >= 15) out.push(w > 0 ? "Warmer, more orange tones" : "Cooler, more blue tones");
  if (g("Highlights2012") <= -30 && (u.Highlights2012 ?? 0) < 25) out.push("Bright areas pulled down");
  if (g("Shadows2012") >= 30 && (u.Shadows2012 ?? 0) < 25) out.push("Dark areas lifted");
  if (g("PostCropVignetteAmount") <= -15) out.push("Corners darkened");
  return out.length ? out : ["Looks close to unedited"];
}

const useMounted = (dep) => { const [m, set] = useState(false);
  useEffect(() => { set(false); const t = requestAnimationFrame(() => requestAnimationFrame(() => set(true))); return () => cancelAnimationFrame(t); }, [dep]);
  return m; };

function Gauge({ value, k }) {
  const C = 2 * Math.PI * 52, m = useMounted(k);
  return (<svg className="gauge" viewBox="0 0 120 120" role="img" aria-label={`Confidence ${Math.round(value * 100)} percent`}>
    <circle cx="60" cy="60" r="52" className="track" />
    <circle cx="60" cy="60" r="52" className="arc" strokeDasharray={C} strokeDashoffset={m ? C * (1 - value) : C} transform="rotate(-90 60 60)" />
    <text x="60" y="58" textAnchor="middle" className="g-num">{Math.round(value * 100)}</text>
    <text x="60" y="76" textAnchor="middle" className="g-sub">percent sure</text></svg>);
}

function Compare({ edited, original }) {
  const [x, setX] = useState(50);
  return (<div className="stage"><div className="frame"><img src={edited} alt="Edited photo" />
    {original && <img className="over" src={original} alt="Original photo" style={{ clipPath: `inset(0 ${100 - x}% 0 0)` }} />}
    {original && <span className="split" style={{ left: `${x}%` }} />}</div>
    {original ? (<><input type="range" min="0" max="100" value={x} onChange={(e) => setX(+e.target.value)} aria-label="Compare original and edited" />
      <p className="cap">Original on the left, edited on the right.</p></>) : <p className="cap">Add the original photo for a far more accurate answer.</p>}</div>);
}

function Ruler({ params, unc = {}, k }) {
  const m = useMounted(k);
  const rows = Object.entries(params).filter(([key]) => key !== "Temperature" && key in LABEL);
  return (<div className="rows">{rows.map(([key, v]) => { const [name, max] = LABEL[key], n = Number(v);
    const w = Math.min(50, (Math.abs(n) / max) * 50);
    return (<div className="row" key={key}><span>{name}</span>
      <span className="ruler"><i className={n < 0 ? "neg" : ""} style={{ width: m ? `${w}%` : 0 }} /></span><b>{fmt(key, v)}{unc[key] ? <small> ±{unc[key]}</small> : null}</b></div>); })}
    {params.Temperature !== undefined && <div className="row"><span>White balance</span><span /><b>{fmt("Temperature", params.Temperature)}</b></div>}</div>);
}

function Curves({ curves }) {
  const line = (a) => a.map((v, i) => `${i},${255 - v}`).join(" ");
  return (<svg className="curves" viewBox="0 0 255 255" role="img" aria-label="Tone curves">
    {[64, 128, 192].map((t) => <path key={t} d={`M${t} 0V255M0 ${t}H255`} className="grid" />)}
    <path d="M0 255L255 0" className="grid diag" />
    {["r", "g", "b", "luma"].map((c) => <polyline key={c} points={line(curves[c])} className={`c-${c}`} />)}</svg>);
}

function ExportPane({ exports }) {
  const [t, setT] = useState("xmp"), [done, setDone] = useState(false);
  const text = t === "xmp" ? exports.xmp : exports.cube || "", ext = t === "xmp" ? "xmp" : "cube";
  const copy = async () => { try { await navigator.clipboard.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); } catch {} };
  const save = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text]));
    a.download = `detected-edit.${ext}`; a.click(); URL.revokeObjectURL(a.href); };
  return (<div><div className="seg" role="tablist">
    <button className={t === "xmp" ? "on" : ""} onClick={() => setT("xmp")}>Lightroom preset</button>
    {exports.cube && <button className={t === "cube" ? "on" : ""} onClick={() => setT("cube")}>3D LUT</button>}</div>
    <textarea readOnly value={text} aria-label="Export text" />
    <div className="acts"><button className="solid" onClick={copy}>{done ? "Copied" : "Copy"}</button><button onClick={save}>Save file</button></div></div>);
}

export default function App() {
  const [edited, setEdited] = useState(null), [orig, setOrig] = useState(null), [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false), [err, setErr] = useState(""), [tab, setTab] = useState("changes");
  const [url, setUrl] = useState(""), [ourl, setOurl] = useState(""), [over, setOver] = useState(false);
  const eu = useMemo(() => edited && URL.createObjectURL(edited), [edited]);
  const ou = useMemo(() => orig && URL.createObjectURL(orig), [orig]);

  useEffect(() => { if (!edited) return; let live = true; setBusy(true); setErr("");
    analyze(edited, orig).then((r) => live && (setRes(r), setTab("changes"))).catch((e) => live && setErr(e.message)).finally(() => live && setBusy(false));
    return () => { live = false; }; }, [edited, orig]);
  useEffect(() => { const h = (e) => { const f = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/")); if (f) setEdited(f); };
    addEventListener("paste", h); return () => removeEventListener("paste", h); }, []);

  const loadUrls = async (e) => { e.preventDefault(); setErr("");
    try { if (ourl.trim()) setOrig(await fetchImage(ourl.trim())); if (url.trim()) setEdited(await fetchImage(url.trim())); }
    catch (x) { setErr(x.message); } };
  const reset = () => { setEdited(null); setOrig(null); setRes(null); setErr(""); setUrl(""); setOurl(""); };
  const head = res && (res.layer === "fitted" && res.unrecognized_pipeline ? "Partly explained" : HEAD[res.layer]);
  const k = res ? JSON.stringify(res.params) : "";

  return (<main>
    <header><h1>How was this photo edited?</h1>{edited && <button className="ghost" onClick={reset}>Start over</button>}</header>
    {!edited && (<section className="start">
      <label className={`drop${over ? " over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) setEdited(f); }}>
        <input type="file" accept="image/*" onChange={(e) => e.target.files[0] && setEdited(e.target.files[0])} />
        <strong>Drop an edited photo here</strong><span>Choose a file, paste with Ctrl+V, or use a link below.</span></label>
      <form onSubmit={loadUrls}><input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="Link to the edited photo" aria-label="Edited photo link" />
        <input value={ourl} onChange={(e) => setOurl(e.target.value)} placeholder="Link to the original (optional)" aria-label="Original photo link" />
        <button className="solid" disabled={!url.trim() && !ourl.trim()}>Load from links</button></form>
      <p className="hint">Photos are analysed on your computer and never stored.</p></section>)}
    {err && <p className="err" role="alert">{err}</p>}
    {edited && (<section className="work"><div><Compare edited={eu} original={ou} />
      <label className="add">{orig ? `Original: ${orig.name}` : "Add the original photo"}<input type="file" accept="image/*" onChange={(e) => setOrig(e.target.files[0] || null)} /></label></div>
      <div className="panel">{busy && <p className="busy">Looking closely…</p>}
        {res && !busy && (<>
          <ol className="steps">{STEPS.map(([id, name, sub]) => <li key={id} className={res.layer === id ? "on" : ""}><b>{name}</b><span>{sub}</span></li>)}</ol>
          <div className="verdict"><Gauge value={res.confidence} k={k} /><div><h2>{head}</h2>
            <p>{res.exact ? "Read straight from the file, so nothing is guessed." : res.layer === "fitted" ? (res.calibrated ? "Fitted to your original and corrected against real Lightroom exports." : "Fitted to your original with our model of the sliders: close to Lightroom, not exact.") : "Estimated from the pixels alone, so treat the numbers as rough."}</p></div></div>
          {res.warnings?.map((w) => <p className="warn" key={w}>{w}</p>)}
          <div className="seg" role="tablist">{[["changes", "What changed"], ["numbers", "Numbers"], ["curves", "Curves"], ["export", "Export"]]
            .filter(([id]) => id !== "curves" || res.curves).map(([id, n]) => <button key={id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>{n}</button>)}</div>
          {tab === "changes" && <ul className="changes">{sentences(res.params, res.uncertainty).map((s) => <li key={s}>{s}</li>)}</ul>}
          {tab === "numbers" && <Ruler params={res.params} unc={res.uncertainty} k={k} />}
          {tab === "curves" && res.curves && <Curves curves={res.curves} />}
          {tab === "export" && <ExportPane exports={res.exports} />}</>)}</div></section>)}
  </main>);
}
