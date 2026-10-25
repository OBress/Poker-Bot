// What the demo server sends (demo/server/table.py Table.view, main.py, history.py).

export type SeatKind = "empty" | "human" | "pokerbot" | "bot" | "tag";
export type Mode = "cash" | "read";

export interface Read {
  style: string;
  confidence: number;
  hands: number;
  probs: Record<string, number>; // every style, most likely first
}

export interface Stat {
  key: string;
  label: string;
  n: number; // opportunities seen
  x: number; // times it happened
  rate: number | null;
  styles: Record<string, number>; // each style's average rate
}

export interface ReadDetail extends Read {
  stats: Stat[];
  trajectory: { hand: number; probs: Record<string, number> }[];
  about: string;
}

export interface SeatView {
  index: number;
  kind: SeatKind;
  name: string;
  human: boolean;
  occupied: boolean;
  you: boolean;
  stack: number;
  bet: number;
  inHand: boolean;
  folded: boolean;
  allIn: boolean;
  cards: string[] | null; // face-up cards (yours, or shown at showdown)
  hasCards: boolean;
  lastAction: string;
  net: number;
  hands: number;
  read: Read | null; // PokerBot's live read of this player
  style: string | null; // a bot's true style
  engineMode: string | null;
}

export interface Legal {
  toCall: number;
  canRaise: boolean;
  minRaiseTo: number;
  maxRaiseTo: number;
  bet: number;
  pot: number;
}

export interface Results {
  shown: Record<string, string[]>;
  mucked: number[];
  winners: { seat: number; amount: number }[];
}

export interface GameInfo {
  id: string;
  mode: Mode;
  target: number | null;
  hands: number;
  finished: boolean;
  started: number;
}

interface Common {
  type: "state";
  deadline: number | null;
  botDelay: number;
  connected: number;
  version: number;
  error?: string;
}

/** On the home screen nobody is playing: ``game`` is null. */
export type ServerView = (Common & { game: null; you: null }) | TableView;

export interface TableView extends Common {
  game: GameInfo;
  running: boolean;
  handNo: number;
  seats: SeatView[];
  button: number;
  board: string[];
  pot: number;
  street: string | null;
  actor: number | null;
  you: number | null;
  legal: Legal | null;
  results: Results | null;
  log: string[];
  bigBlind: number;
  styles: string[];
  myRead: ReadDetail | null;
  finalReads: Record<string, ReadDetail> | null;
}

export interface HistorySummary {
  id: string;
  mode: Mode;
  target: number | null;
  hands: number;
  finished: boolean;
  started: number;
  ended: number | null;
  bigBlind: number;
  humans: { name: string; net: number; read: string | null; confidence: number | null }[];
  pokerbots: { name: string; net: number }[];
}

export interface GameRecord extends GameInfo {
  ended: number | null;
  bigBlind: number;
  players: {
    seat: number;
    name: string;
    kind: SeatKind;
    trueStyle: string | null;
    net: number;
    hands: number;
    read: Read | null;
    stats: Stat[] | null;
    trajectory: { hand: number; probs: Record<string, number> }[] | null;
  }[];
  styleInfo: Record<string, string>;
  log: string[];
}

export type ClientMessage =
  | { type: "newGame"; mode: Mode; hands?: number; name: string }
  | { type: "endGame" }
  | { type: "sit"; seat: number; name: string }
  | { type: "leave" }
  | { type: "configure"; seat: number; kind: SeatKind; style?: number | null }
  | { type: "start" }
  | { type: "pause" }
  | { type: "speed"; botDelay: number }
  | { type: "action"; action: "fold" | "check" | "call" | "raise"; amount?: number };
