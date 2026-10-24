// The table view the demo server sends (demo/server/table.py, Table.view).

export type SeatKind = "empty" | "human" | "pokerbot" | "bot" | "tag";

export interface Read {
  style: string;
  confidence: number;
  hands: number;
  top: { style: string; p: number }[];
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
  read: Read | null; // PokerBot's live read of this player
  style: string | null; // a bot's true style (hidden unless "reveal" is on)
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

export interface TableView {
  type: "state";
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
  deadline: number | null;
  botDelay: number;
  connected: number;
  error?: string;
}

export type ClientMessage =
  | { type: "sit"; seat: number; name: string }
  | { type: "leave" }
  | { type: "configure"; seat: number; kind: SeatKind; style?: number | null }
  | { type: "start" }
  | { type: "pause" }
  | { type: "speed"; botDelay: number }
  | { type: "action"; action: "fold" | "check" | "call" | "raise"; amount?: number };
