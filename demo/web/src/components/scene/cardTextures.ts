import * as THREE from "three";

// Card faces and the back are drawn on canvases once and cached as textures.

const W = 256;
const H = 358;
const SUITS: Record<string, { glyph: string; color: string }> = {
  c: { glyph: "♣", color: "#111827" },
  d: { glyph: "♦", color: "#d4202c" },
  h: { glyph: "♥", color: "#d4202c" },
  s: { glyph: "♠", color: "#111827" },
};
const cache = new Map<string, THREE.Texture>();

function roundedRect(ctx: CanvasRenderingContext2D, r: number) {
  ctx.beginPath();
  ctx.moveTo(r, 0);
  ctx.arcTo(W, 0, W, H, r);
  ctx.arcTo(W, H, 0, H, r);
  ctx.arcTo(0, H, 0, 0, r);
  ctx.arcTo(0, 0, W, 0, r);
  ctx.closePath();
}

function texture(key: string, draw: (ctx: CanvasRenderingContext2D) => void): THREE.Texture {
  const hit = cache.get(key);
  if (hit) return hit;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  draw(ctx);
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  cache.set(key, t);
  return t;
}

/** "As" -> a face texture. */
export function faceTexture(card: string): THREE.Texture {
  return texture(`face-${card}`, (ctx) => {
    const rank = card[0] === "T" ? "10" : card[0];
    const suit = SUITS[card[1]];
    roundedRect(ctx, 22);
    ctx.fillStyle = "#fbfaf6";
    ctx.fill();
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#d6d3cb";
    ctx.stroke();
    ctx.fillStyle = suit.color;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "bold 74px Georgia, serif";
    ctx.fillText(rank, 52, 58);
    ctx.font = "60px Georgia, serif";
    ctx.fillText(suit.glyph, 52, 124);
    ctx.font = "170px Georgia, serif";
    ctx.fillText(suit.glyph, W / 2 + 22, H / 2 + 40);
  });
}

export function backTexture(): THREE.Texture {
  return texture("back", (ctx) => {
    roundedRect(ctx, 22);
    ctx.fillStyle = "#f4f1ea";
    ctx.fill();
    ctx.save();
    ctx.translate(16, 16);
    const w = W - 32;
    const h = H - 32;
    ctx.fillStyle = "#1f3b73";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 3;
    for (let i = -h; i < w; i += 18) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i + h, h);
      ctx.moveTo(i + h, 0);
      ctx.lineTo(i, h);
      ctx.stroke();
    }
    ctx.strokeStyle = "#e9c46a";
    ctx.lineWidth = 5;
    ctx.strokeRect(10, 10, w - 20, h - 20);
    ctx.fillStyle = "#e9c46a";
    ctx.font = "bold 44px Georgia, serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("PB", w / 2, h / 2);
    ctx.restore();
  });
}

export const CARD_W = 0.42;
export const CARD_H = (CARD_W * H) / W;
