"use client";

import { useEffect, type ReactNode } from "react";

/** A centered dialog; Escape or a click outside closes it. */
export function Modal({ title, onClose, children, wide = false, footer }: {
  title: ReactNode; onClose: () => void; children: ReactNode; wide?: boolean; footer?: ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm" onClick={onClose}>
      <div
        className={`flex max-h-[90vh] w-full ${wide ? "max-w-3xl" : "max-w-xl"} flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#17121f] shadow-2xl`}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-3.5">
          <div className="text-base font-semibold">{title}</div>
          <button className="rounded-md px-2 py-1 text-white/60 hover:bg-white/10 hover:text-white" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-5">{children}</div>
        {footer && <div className="flex justify-end gap-2 border-t border-white/10 px-5 py-3">{footer}</div>}
      </div>
    </div>
  );
}

export function Logo({ small = false }: { small?: boolean }) {
  return (
    <span className={`font-bold tracking-tight ${small ? "text-lg" : "text-4xl sm:text-5xl"}`}>
      Poker<span className="text-teal-300">Bot</span>
    </span>
  );
}

export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "teal" | "amber" }) {
  const tones = {
    neutral: "bg-white/10 text-white/75",
    teal: "bg-teal-500/20 text-teal-200",
    amber: "bg-amber-500/20 text-amber-200",
  };
  return <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${tones[tone]}`}>{children}</span>;
}
