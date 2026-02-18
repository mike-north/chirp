/**
 * Chirp Daemon - Claude Code Text Refiner
 *
 * This daemon acts as a sidecar Node.js process for the Chirp macOS app.
 * It receives voice transcription text from the app and sends it to the
 * Anthropic API via the Claude Code SDK for grammar and punctuation cleanup.
 *
 * Protocol:
 * - Communication uses newline-delimited JSON over a Unix domain socket
 * - Socket location: /tmp/chirp-claude.sock
 * - Socket permissions: 0600 (owner-only read/write)
 *
 * Request format:
 * - {"action": "ping"} - Health check
 * - {"action": "refine", "text": "...", "systemPrompt": "...", "model": "..."} - Refine text
 *
 * Response format:
 * - {"ok": true, "refined": "..."} - Success (refined is omitted for ping)
 * - {"ok": false, "error": "..."} - Error
 */

import * as net from "node:net";
import * as fs from "node:fs";
import { query } from "@anthropic-ai/claude-code";

const SOCKET_PATH = "/tmp/chirp-claude.sock";

interface PingRequest {
  action: "ping";
}

interface RefineRequest {
  action: "refine";
  text: string;
  systemPrompt: string;
  model: string;
}

type Request = PingRequest | RefineRequest;

interface SuccessResponse {
  ok: true;
  refined?: string;
}

interface ErrorResponse {
  ok: false;
  error: string;
}

type Response = SuccessResponse | ErrorResponse;

function cleanupSocket(): void {
  try {
    fs.unlinkSync(SOCKET_PATH);
  } catch {
    // Socket file doesn't exist, nothing to clean up
  }
}

/**
 * Handles a refine request by calling the Claude Code SDK query() function.
 *
 * Invokes the SDK with the provided text, system prompt, and model to get
 * a cleaned-up version of the transcription text (corrected grammar,
 * punctuation, capitalization, etc.).
 *
 * @param request - The refine request containing text, systemPrompt, and model
 * @returns A response with the refined text or an error
 */
async function handleRefine(request: RefineRequest): Promise<Response> {
  try {
    let lastResult = "";
    const conversation = query({
      prompt: request.text,
      options: {
        customSystemPrompt: request.systemPrompt,
        allowedTools: [],
        maxTurns: 1,
        model: request.model,
        permissionMode: "bypassPermissions",
      },
    });

    for await (const message of conversation) {
      if (
        message.type === "result" &&
        message.subtype === "success" &&
        "result" in message
      ) {
        lastResult = message.result;
      }
    }

    return { ok: true, refined: lastResult };
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    return { ok: false, error: errorMessage };
  }
}

/**
 * Dispatches incoming requests by action type.
 *
 * Routes to the appropriate handler based on the action field:
 * - "ping" returns an immediate success response
 * - "refine" delegates to handleRefine() for Claude Code SDK processing
 *
 * @param data - The incoming request object
 * @returns A response (synchronous for ping, async for refine)
 */
function handleRequest(data: Request): Promise<Response> | Response {
  switch (data.action) {
    case "ping":
      return { ok: true };
    case "refine":
      return handleRefine(data);
    default:
      return { ok: false, error: `Unknown action: ${(data as Record<string, unknown>).action}` };
  }
}

// Unix domain socket server. Each connection uses a buffered line protocol:
// incoming data is accumulated in a buffer, split on newlines, and each
// complete line is parsed as a JSON request. Responses are written back
// as JSON followed by a newline.
const server = net.createServer((connection) => {
  let buffer = "";

  connection.on("data", (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop()!;

    for (const line of lines) {
      if (line.trim() === "") continue;

      let request: Request;
      try {
        request = JSON.parse(line) as Request;
      } catch {
        const response: ErrorResponse = { ok: false, error: "Invalid JSON" };
        connection.write(JSON.stringify(response) + "\n");
        continue;
      }

      const result = handleRequest(request);
      void Promise.resolve(result).then((response) => {
        connection.write(JSON.stringify(response) + "\n");
      });
    }
  });
});

cleanupSocket();

server.listen(SOCKET_PATH, () => {
  fs.chmodSync(SOCKET_PATH, 0o600);
  console.error(`Chirp daemon listening on ${SOCKET_PATH}`);
});

function shutdown(): void {
  server.close();
  cleanupSocket();
  process.exit(0);
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
