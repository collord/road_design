// A fine route's long profile: ground and road design elevation against distance, with hairpins,
// bends and fill crossings marked. Plain SVG, colored from the page's tokens.
import type { FineOut } from "./messages";

const W = 640, H = 230;
const M = { left: 48, right: 12, top: 12, bottom: 32 };

/** Cumulative distance (m) along the route's points. */
export function stations(f: FineOut): number[] {
  const d = [0];
  for (let i = 1; i < f.x.length; i++) d.push(d[i - 1] + Math.hypot(f.x[i] - f.x[i - 1], f.y[i] - f.y[i - 1]));
  return d;
}

/** Round step for about `n` ticks over a span. */
function tickStep(span: number, n: number): number {
  const raw = span / n, p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? 10 * p;
}

const MARKS: Record<string, string> = { hairpin: "var(--route)", bend: "var(--alt)", fill_crossing: "var(--fine)" };

export function profileSvg(f: FineOut): string {
  const d = stations(f), L = d[d.length - 1] || 1;
  const zs = [...f.z, ...f.ground].filter(Number.isFinite);
  let zmin = Infinity, zmax = -Infinity;
  for (const v of zs) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
  const pad = Math.max(5, 0.05 * (zmax - zmin));
  zmin -= pad; zmax += pad;
  const px = (s: number) => M.left + (s / L) * (W - M.left - M.right);
  const py = (z: number) => H - M.bottom - ((z - zmin) / (zmax - zmin)) * (H - M.top - M.bottom);
  const line = (zz: number[]) => zz.map((z, i) => `${i ? "L" : "M"}${px(d[i]).toFixed(1)},${py(z).toFixed(1)}`).join("");
  const parts: string[] = [];
  // Macro spans behind the lines.
  f.segments.forEach((s, i) => {
    const color = MARKS[s.kind];
    if (!color) return;
    const a = d[s.first], b = d[i + 1 < f.segments.length ? f.segments[i + 1].first : d.length - 1];
    parts.push(`<rect x="${px(a).toFixed(1)}" y="${M.top}" width="${Math.max(1, px(b) - px(a)).toFixed(1)}" ` +
               `height="${H - M.top - M.bottom}" fill="${color}" opacity="0.18"><title>${s.kind.replace("_", " ")}: ` +
               `${s.length.toFixed(0)} m, grade ${(100 * s.grade).toFixed(1)}%</title></rect>`);
  });
  // Axes and ticks.
  const kmStep = tickStep(L / 1000, 6), zStep = tickStep(zmax - zmin, 5);
  for (let k = 0; k * kmStep * 1000 <= L + 1e-6; k++) {
    const x = px(k * kmStep * 1000);
    parts.push(`<line x1="${x}" x2="${x}" y1="${H - M.bottom}" y2="${H - M.bottom + 4}" stroke="var(--muted)"/>` +
               `<text x="${x}" y="${H - M.bottom + 16}" text-anchor="middle">${+(k * kmStep).toFixed(2)}</text>`);
  }
  for (let z = Math.ceil(zmin / zStep) * zStep; z <= zmax; z += zStep) {
    const y = py(z);
    parts.push(`<line x1="${M.left}" x2="${W - M.right}" y1="${y}" y2="${y}" stroke="var(--rule)"/>` +
               `<text x="${M.left - 6}" y="${y + 4}" text-anchor="end">${Math.round(z)}</text>`);
  }
  parts.push(`<text x="${(M.left + W - M.right) / 2}" y="${H - 4}" text-anchor="middle">km</text>`);
  parts.push(`<path d="${line(f.ground)}" fill="none" stroke="var(--muted)" stroke-width="1.2"/>`);
  parts.push(`<path d="${line(f.z)}" fill="none" stroke="var(--accent)" stroke-width="2"/>`);
  return `<svg viewBox="0 0 ${W} ${H}" class="profile" role="img" aria-label="Long profile of corridor ${f.index + 1}">` +
         `${parts.join("")}</svg>`;
}

/** Summary line under the profile: climb, steepest segment, and how much of the road leaves the ground. */
export function profileSummary(f: FineOut): string {
  let up = 0, down = 0, offGround = 0;
  const d = stations(f);
  for (let i = 1; i < f.z.length; i++) {
    const dz = f.z[i] - f.z[i - 1];
    if (dz > 0) up += dz; else down -= dz;
    if (Math.abs(f.z[i] - f.ground[i]) > 1) offGround += d[i] - d[i - 1];
  }
  const steepest = f.segments.reduce((m, s) => Math.max(m, Math.abs(s.grade)), 0);
  return `corridor ${f.index + 1}: rise ${Math.round(up)} m, fall ${Math.round(down)} m, steepest segment ` +
         `${(100 * steepest).toFixed(1)}%; design more than 1 m off the ground for ${(offGround / 1000).toFixed(2)} km`;
}
