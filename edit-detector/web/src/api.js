const BASE = "/api/v1";
const fail = async (r, fallback) => {
  const j = await r.json().catch(() => ({}));
  throw new Error(j.error?.message || `${fallback} (${r.status})`);
};
export async function analyze(edited, original) {
  const f = new FormData();
  f.append("edited", edited);
  if (original) f.append("original", original);
  const r = await fetch(`${BASE}/analyze`, { method: "POST", body: f });
  if (!r.ok) await fail(r, "Analysis failed");
  return r.json();
}
export async function fetchImage(url) {
  const r = await fetch(`${BASE}/fetch?url=${encodeURIComponent(url)}`);
  if (!r.ok) await fail(r, "Could not load that link");
  const b = await r.blob();
  const name = decodeURIComponent(url.split("?")[0].split("/").pop()) || "image";
  return new File([b], name, { type: b.type });
}
