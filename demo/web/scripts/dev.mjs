// Development: the Python game server and `next dev` (hot reload) together: `npm run dev:all`.
// Open http://localhost:3000. For playing, `npm run demo` serves everything from port 8000.
import { spawn } from "node:child_process";

const procs = [
  spawn("npm", ["run", "server"], { stdio: "inherit", shell: true }),
  spawn("npm", ["run", "dev"], { stdio: "inherit", shell: true }),
];
const stop = () => procs.forEach((p) => p.kill());
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
procs.forEach((p) => p.on("exit", (code) => { if (code) { stop(); process.exit(code); } }));
