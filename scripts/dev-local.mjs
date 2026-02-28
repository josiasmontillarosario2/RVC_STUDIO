import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");
const rvcDir = resolve(repoRoot, "rvc_minimal");

const PYTHON_CMD = process.env.RVC_PYTHON || "python";
const RVC_PORT = process.env.RVC_PORT || "8000";
const NEXT_PORT = process.env.PORT || "3000";
const RVC_API_BASE_URL = process.env.RVC_API_BASE_URL || `http://127.0.0.1:${RVC_PORT}`;

let shuttingDown = false;
const children = [];

function startProcess(name, command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd || repoRoot,
    env: { ...process.env, ...options.env },
    shell: process.platform === "win32",
    stdio: "pipe",
  });

  child.stdout.on("data", (chunk) => {
    process.stdout.write(`[${name}] ${chunk}`);
  });

  child.stderr.on("data", (chunk) => {
    process.stderr.write(`[${name}] ${chunk}`);
  });

  child.on("exit", (code, signal) => {
    process.stdout.write(`\n[${name}] exited (code=${code ?? "null"} signal=${signal ?? "null"})\n`);
    if (!shuttingDown) {
      shutdown(code ?? 1);
    }
  });

  child.on("error", (err) => {
    process.stderr.write(`\n[${name}] failed to start: ${err.message}\n`);
    if (!shuttingDown) {
      shutdown(1);
    }
  });

  children.push(child);
  return child;
}

function shutdown(exitCode = 0) {
  if (shuttingDown) return;
  shuttingDown = true;

  for (const child of children) {
    if (!child.killed) {
      try {
        if (process.platform === "win32") {
          child.kill("SIGTERM");
        } else {
          child.kill("SIGTERM");
        }
      } catch {
        // ignore shutdown race conditions
      }
    }
  }

  setTimeout(() => process.exit(exitCode), 300);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

process.stdout.write(
  [
    "Starting local stack:",
    `- Next.js: http://127.0.0.1:${NEXT_PORT}`,
    `- RVC API: ${RVC_API_BASE_URL}`,
    `- Python command: ${PYTHON_CMD}`,
    "",
  ].join("\n")
);

startProcess(
  "rvc",
  PYTHON_CMD,
  ["-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", RVC_PORT, "--reload"],
  { cwd: rvcDir }
);

startProcess(
  "next",
  "npx",
  ["next", "dev", "--port", NEXT_PORT],
  {
    cwd: repoRoot,
    env: {
      RVC_API_BASE_URL,
    },
  }
);
