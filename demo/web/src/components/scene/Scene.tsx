"use client";

import { ContactShadows, Html, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as THREE from "three";
import type { SeatView, TableView } from "@/lib/types";
import { backTexture, CARD_H, CARD_W, faceTexture } from "./cardTextures";

// Table geometry (world units): an elliptical felt of radii RX x RZ at y = 0.
const RX = 3.6;
const RZ = 2.05;
const SEATS = 8;
const DECK: [number, number, number] = [0, 0.8, -0.4];

type V3 = [number, number, number];

/** Point on the ellipse scaled by (fx, fz) for display slot k (0 = front, toward the camera). */
function around(k: number, fx: number, fz: number, y = 0): V3 {
  const a = Math.PI / 2 + (k * 2 * Math.PI) / SEATS;
  return [RX * fx * Math.cos(a), y, RZ * fz * Math.sin(a)];
}

/** Moves its children smoothly to a target position and rotation; starts at ``from``. */
function Smooth({ to, rot = [0, 0, 0], from, speed = 7, children }: {
  to: V3; rot?: V3; from?: V3; speed?: number; children: ReactNode;
}) {
  const ref = useRef<THREE.Group>(null);
  const init = useRef(false);
  const target = useMemo(() => new THREE.Quaternion(), []);
  useFrame((_, dt) => {
    const g = ref.current;
    if (!g) return;
    if (!init.current) {
      g.position.set(...(from ?? to));
      g.rotation.set(...rot);
      init.current = true;
    }
    const t = 1 - Math.exp(-speed * dt);
    g.position.lerp(new THREE.Vector3(...to), t);
    target.setFromEuler(new THREE.Euler(...rot));
    g.quaternion.slerp(target, t);
  });
  return <group ref={ref}>{children}</group>;
}

function Card({ card, faceUp }: { card: string | null; faceUp: boolean }) {
  const face = useMemo(() => (card ? faceTexture(card) : backTexture()), [card]);
  const back = useMemo(() => backTexture(), []);
  // a card lying flat; flipping is a rotation of the parent
  return (
    <group rotation={[faceUp ? -Math.PI / 2 : Math.PI / 2, 0, 0]}>
      <mesh castShadow>
        <planeGeometry args={[CARD_W, CARD_H]} />
        <meshStandardMaterial map={face} roughness={0.55} side={THREE.FrontSide} />
      </mesh>
      <mesh rotation={[0, Math.PI, 0]}>
        <planeGeometry args={[CARD_W, CARD_H]} />
        <meshStandardMaterial map={back} roughness={0.6} side={THREE.FrontSide} />
      </mesh>
    </group>
  );
}

const DENOMS = [
  { v: 10000, color: "#f59e0b" },
  { v: 2500, color: "#7c3aed" },
  { v: 500, color: "#111827" },
  { v: 100, color: "#16a34a" },
  { v: 25, color: "#2563eb" },
];

function Chips({ amount, label, bb }: { amount: number; label?: boolean; bb: number }) {
  const stacks = useMemo(() => {
    let left = amount;
    const out: { color: string; n: number }[] = [];
    for (const d of DENOMS) {
      const n = Math.floor(left / d.v);
      if (n > 0) out.push({ color: d.color, n: Math.min(n, 12) });
      left -= n * d.v;
    }
    if (left > 0 && out.length === 0) out.push({ color: "#dc2626", n: 1 });
    return out.slice(0, 4);
  }, [amount]);
  // the label stays mounted (an Html root unmounting mid-render upsets React); it just empties
  return (
    <group>
      {amount > 0 && stacks.map((s, i) => (
        <group key={i} position={[(i - (stacks.length - 1) / 2) * 0.17, 0, (i % 2) * 0.05]}>
          {Array.from({ length: s.n }, (_, j) => (
            <mesh key={j} position={[0, 0.012 + j * 0.022, 0]} castShadow>
              <cylinderGeometry args={[0.075, 0.075, 0.02, 24]} />
              <meshStandardMaterial color={s.color} roughness={0.4} metalness={0.1} />
            </mesh>
          ))}
          <mesh position={[0, 0.013 + (s.n - 1) * 0.022 + 0.011, 0]}>
            <cylinderGeometry args={[0.05, 0.05, 0.002, 24]} />
            <meshStandardMaterial color="#f8fafc" />
          </mesh>
        </group>
      ))}
      {label && (
        <Html position={[0, 0.35, 0]} center zIndexRange={[10, 0]} className="pointer-events-none select-none">
          {amount > 0 && (
            <div className="rounded-full bg-black/70 px-2 py-0.5 text-xs font-semibold whitespace-nowrap text-amber-200">
              {fmtBB(amount, bb)}
            </div>
          )}
        </Html>
      )}
    </group>
  );
}

export function fmtBB(chips: number, bb: number) {
  const v = chips / bb;
  return `${Number.isInteger(v) ? v : v.toFixed(1)} bb`;
}

function feltTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 1024;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(512, 512, 60, 512, 512, 560);
  g.addColorStop(0, "#1f7a4f");
  g.addColorStop(1, "#0d4a2e");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 1024, 1024);
  ctx.globalAlpha = 0.12;
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 92px Georgia, serif";
  ctx.textAlign = "center";
  ctx.fillText("POKERBOT", 512, 720);
  ctx.globalAlpha = 0.18;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.ellipse(512, 512, 330, 330, 0, 0, Math.PI * 2);
  ctx.stroke();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function TableModel() {
  const felt = useMemo(() => feltTexture(), []);
  const rail = useMemo(() => {
    const s = new THREE.Shape();
    s.absellipse(0, 0, RX + 0.42, RZ + 0.42, 0, Math.PI * 2, false, 0);
    const hole = new THREE.Path();
    hole.absellipse(0, 0, RX, RZ, 0, Math.PI * 2, true, 0);
    s.holes.push(hole);
    return new THREE.ExtrudeGeometry(s, {
      depth: 0.12, bevelEnabled: true, bevelThickness: 0.07, bevelSize: 0.08, bevelSegments: 6,
      curveSegments: 96,
    });
  }, []);
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow scale={[RX, RZ, 1]}>
        <circleGeometry args={[1, 96]} />
        <meshStandardMaterial map={felt} roughness={0.95} />
      </mesh>
      <mesh geometry={rail} rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]} castShadow>
        <meshStandardMaterial color="#3b2416" roughness={0.45} metalness={0.05} />
      </mesh>
      <mesh position={[0, -0.18, 0]} scale={[RX + 0.45, 1, RZ + 0.45]}>
        <cylinderGeometry args={[1, 0.92, 0.25, 96]} />
        <meshStandardMaterial color="#24150d" roughness={0.6} />
      </mesh>
      <mesh position={[0, -1.0, 0]}>
        <cylinderGeometry args={[0.55, 0.9, 1.4, 32]} />
        <meshStandardMaterial color="#1a1110" roughness={0.7} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.7, 0]} receiveShadow>
        <circleGeometry args={[14, 64]} />
        <meshStandardMaterial color="#2a1d2e" roughness={1} />
      </mesh>
    </group>
  );
}

const PALETTE = ["#e76f51", "#2a9d8f", "#e9c46a", "#8ab17d", "#b56576", "#6d597a", "#f4a261", "#457b9d"];

function Avatar({ seat, active, winner }: { seat: SeatView; active: boolean; winner: boolean }) {
  const ring = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (ring.current) {
      const m = ring.current.material as THREE.MeshStandardMaterial;
      m.emissiveIntensity = active ? 1.2 + Math.sin(clock.elapsedTime * 5) * 0.8 : winner ? 1.5 : 0;
    }
  });
  const bot = seat.kind === "pokerbot";
  const color = bot ? "#14b8a6" : seat.human ? "#f97316" : seat.kind === "tag" ? "#94a3b8" : PALETTE[seat.index % PALETTE.length];
  const dim = seat.folded ? 0.35 : 1;
  return (
    <group>
      <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.32, 0]}>
        <torusGeometry args={[0.5, 0.04, 12, 48]} />
        <meshStandardMaterial color={winner ? "#fbbf24" : "#38bdf8"} emissive={winner ? "#fbbf24" : "#38bdf8"} emissiveIntensity={0} transparent opacity={active || winner ? 1 : 0} />
      </mesh>
      <mesh position={[0, 0.15, 0]} castShadow>
        <capsuleGeometry args={[0.32, 0.55, 8, 16]} />
        <meshStandardMaterial color={color} roughness={bot ? 0.25 : 0.7} metalness={bot ? 0.6 : 0} transparent opacity={dim} />
      </mesh>
      {bot ? (
        <group position={[0, 0.98, 0]}>
          <mesh castShadow>
            <boxGeometry args={[0.5, 0.42, 0.44]} />
            <meshStandardMaterial color="#0f766e" metalness={0.7} roughness={0.25} transparent opacity={dim} />
          </mesh>
          <mesh position={[0, 0.04, 0.225]}>
            <boxGeometry args={[0.36, 0.1, 0.02]} />
            <meshStandardMaterial color="#5eead4" emissive="#5eead4" emissiveIntensity={2} />
          </mesh>
          <mesh position={[0, 0.3, 0]}>
            <cylinderGeometry args={[0.015, 0.015, 0.18, 8]} />
            <meshStandardMaterial color="#99f6e4" />
          </mesh>
          <mesh position={[0, 0.41, 0]}>
            <sphereGeometry args={[0.045, 12, 12]} />
            <meshStandardMaterial color="#5eead4" emissive="#5eead4" emissiveIntensity={2} />
          </mesh>
        </group>
      ) : (
        <mesh position={[0, 0.98, 0]} castShadow>
          <sphereGeometry args={[0.27, 32, 32]} />
          <meshStandardMaterial color="#f1c7a6" roughness={0.8} transparent opacity={dim} />
        </mesh>
      )}
    </group>
  );
}

function Nameplate({ seat, view, reveal, showReads, active }: { seat: SeatView; view: TableView; reveal: boolean; showReads: boolean; active: boolean }) {
  const bb = view.bigBlind;
  const won = view.results?.winners.find((w) => w.seat === seat.index);
  return (
    <div className={`w-44 rounded-xl border px-2.5 py-1.5 text-center shadow-lg backdrop-blur-sm select-none ${active ? "border-sky-400 bg-sky-950/80" : "border-white/10 bg-black/65"} ${seat.folded ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-center gap-1.5 text-sm font-semibold text-white">
        {seat.kind === "pokerbot" && <span className="rounded bg-teal-500/30 px-1 text-[10px] text-teal-200">ENGINE</span>}
        {seat.you && <span className="rounded bg-orange-500/30 px-1 text-[10px] text-orange-200">YOU</span>}
        <span className="truncate">{seat.name}</span>
      </div>
      <div className="text-xs text-amber-200">{fmtBB(seat.stack, bb)}</div>
      {won ? (
        <div className="mt-0.5 text-xs font-bold text-yellow-300">wins {fmtBB(won.amount, bb)}</div>
      ) : seat.lastAction ? (
        <div className="mt-0.5 text-xs text-sky-200">{seat.lastAction}</div>
      ) : null}
      {active && view.deadline && <Countdown deadline={view.deadline} />}
      {showReads && seat.read && (
        <div className="mt-1 rounded-md bg-teal-900/60 px-1 py-0.5 text-[10px] leading-tight text-teal-100" title={Object.entries(seat.read.probs).slice(0, 3).map(([k, p]) => `${k} ${(p * 100).toFixed(0)}%`).join(", ")}>
          PokerBot reads: <b>{seat.read.style}</b> {(seat.read.confidence * 100).toFixed(0)}%
          <span className="text-teal-300/70"> · {seat.read.hands}h</span>
        </div>
      )}
      {reveal && seat.style && <div className="mt-0.5 text-[10px] text-fuchsia-200">actually: {seat.style}</div>}
      {seat.engineMode && <div className="mt-0.5 text-[10px] text-teal-300/80">{seat.engineMode}</div>}
    </div>
  );
}

/** Seconds left for the human to act (the server folds or checks for them at zero). */
function Countdown({ deadline }: { deadline: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  return <div className="text-[11px] text-sky-300">{Math.max(0, Math.round(deadline - now / 1000))}s</div>;
}

function SeatGroup({ seat, slot, view, reveal, showReads }: { seat: SeatView; slot: number; view: TableView; reveal: boolean; showReads: boolean }) {
  const active = view.actor === seat.index;
  const winner = !!view.results?.winners.some((w) => w.seat === seat.index);
  const pos = around(slot, 1.3, 1.45, 0);
  const face = Math.atan2(-pos[0], -pos[2]); // look at the table centre
  const cardPos = around(slot, 0.8, 0.74, 0.012);
  const betPos = around(slot, 0.58, 0.48, 0);
  const own = seat.you;
  const cards = seat.hasCards || seat.cards ? seat.cards ?? [null, null] : [];
  return (
    <group>
      {seat.occupied && !own && (
        <group position={pos} rotation={[0, face, 0]}>
          <Avatar seat={seat} active={active} winner={winner} />
        </group>
      )}
      <Html position={[pos[0], 1.75, pos[2]]} center zIndexRange={[10, 0]}>
        {own ? null : seat.occupied ? (
          <Nameplate seat={seat} view={view} reveal={reveal} showReads={showReads} active={active} />
        ) : (
          <div className="rounded-lg bg-black/40 px-2 py-1 text-xs text-white/50 select-none">{seat.kind === "human" ? "Open seat" : "Empty"}</div>
        )}
      </Html>
      {cards.map((c, i) => {
        const side = (i - 0.5) * (own ? 0.5 : 0.3);
        const perp: V3 = own ? [side, 0, 0] : [Math.cos(face) * side, 0, -Math.sin(face) * side];
        const lift = own ? 0.32 : 0.012 + i * 0.003;
        const to: V3 = [cardPos[0] + perp[0], lift, cardPos[2] + perp[2] + (own ? 0.35 : 0)];
        // your cards stand up facing the camera, tilted back; others lie on the felt
        const rot: V3 = own ? [-0.75, (0.5 - i) * 0.12, 0] : [0, face + (i - 0.5) * 0.15, 0];
        return (
          <Smooth key={`${view.handNo}-${seat.index}-${i}`} to={to} rot={rot} from={DECK}>
            <group scale={own ? 1.35 : 1}>
              {own ? <OwnCard card={c} /> : <Card card={c} faceUp={!!c} />}
            </group>
          </Smooth>
        );
      })}
      <group position={betPos}>
        <Chips amount={seat.bet} bb={view.bigBlind} label />
      </group>
      {view.button === seat.index && (
        <Smooth to={[betPos[0] + Math.cos(face) * 0.45, 0.02, betPos[2] - Math.sin(face) * 0.45]}>
          <DealerButton />
        </Smooth>
      )}
    </group>
  );
}

function DealerButton() {
  const top = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, 128, 128);
    ctx.fillStyle = "#1e293b";
    ctx.font = "bold 84px Georgia, serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("D", 64, 70);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, []);
  return (
    <group>
      <mesh castShadow>
        <cylinderGeometry args={[0.13, 0.13, 0.04, 32]} />
        <meshStandardMaterial color="#f8fafc" roughness={0.3} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.021, 0]}>
        <circleGeometry args={[0.12, 32]} />
        <meshStandardMaterial map={top} roughness={0.3} />
      </mesh>
    </group>
  );
}

/** Your own cards: always face up, standing tilted toward the camera. */
function OwnCard({ card }: { card: string | null }) {
  const face = useMemo(() => (card ? faceTexture(card) : backTexture()), [card]);
  return (
    <mesh castShadow>
      <planeGeometry args={[CARD_W, CARD_H]} />
      <meshStandardMaterial map={face} roughness={0.5} side={THREE.DoubleSide} />
    </mesh>
  );
}

function Board({ view }: { view: TableView }) {
  return (
    <group>
      {view.board.map((c, i) => (
        <Smooth key={`${view.handNo}-b${i}`} to={[(i - 2) * 0.62, 0.012, -0.3]} from={DECK} rot={[0, 0, 0]}>
          <group scale={1.3}>
            <Card card={c} faceUp />
          </group>
        </Smooth>
      ))}
      <group position={[0, 0, 0.55]}>
        <Chips amount={view.pot - view.seats.reduce((s, x) => s + x.bet, 0)} bb={view.bigBlind} />
      </group>
      <Html position={[0, 0.05, -1.05]} center zIndexRange={[10, 0]} className="pointer-events-none select-none">
        {view.pot > 0 && (
          <div className="rounded-full bg-black/70 px-3 py-1 text-sm font-bold whitespace-nowrap text-amber-200">
            Pot {fmtBB(view.pot, view.bigBlind)}
          </div>
        )}
      </Html>
    </group>
  );
}

export default function Scene({ view, reveal, showReads }: { view: TableView; reveal: boolean; showReads: boolean }) {
  const me = view.you ?? 0;
  return (
    <Canvas shadows camera={{ position: [0, 4.9, 8.3], fov: 48 }} dpr={[1, 2]}>
      <color attach="background" args={["#120d18"]} />
      <fog attach="fog" args={["#120d18", 12, 26]} />
      <ambientLight intensity={0.45} />
      <spotLight position={[0, 9, 1]} angle={0.62} penumbra={0.6} intensity={140} castShadow shadow-mapSize={[2048, 2048]} color="#fff4dd" />
      <pointLight position={[-6, 3, -4]} intensity={18} color="#7c3aed" />
      <pointLight position={[6, 3, -4]} intensity={18} color="#0ea5e9" />
      <TableModel />
      <ContactShadows position={[0, 0.002, 0]} scale={[RX * 2.2, RZ * 2.2]} opacity={0.35} blur={2.2} far={1} />
      {view.seats.map((s) => (
        <SeatGroup key={s.index} seat={s} slot={(s.index - me + SEATS) % SEATS} view={view} reveal={reveal} showReads={showReads} />
      ))}
      <Board view={view} />
      <OrbitControls target={[0, 0, 0.3]} enablePan={false} minDistance={4} maxDistance={13} minPolarAngle={0.25} maxPolarAngle={1.25} />
    </Canvas>
  );
}
