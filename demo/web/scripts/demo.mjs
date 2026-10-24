// Starts the Python game server and the Next.js dev server together: `npm run demo`.
import { spawn } from "node:child_process";

const procs = [
  spawn("npm", ["run", "server"], { stdio: "inherit", shell: true }),
  spawn("npm", ["run", "dev"], { stdio: "inherit", shell: true }),
];
const stop = () => procs.forEach((p) => p.kill());
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
procs.forEach((p) => p.on("exit", (code) => { if (code) { stop(); process.exit(code); } }));
