# Claude Text Refinement

## Overview

Chirp integrates with Claude to automatically refine transcribed text before typing it into the active application. This feature uses a multi-process architecture to leverage the Claude Code SDK for high-quality text transformation.

## Architecture

The Claude refinement feature uses a layered architecture:

```
Swift App (Chirp.app)
    ↓
Unix Domain Socket (/tmp/chirp-claude.sock)
    ↓
Node.js Daemon (chirp-daemon)
    ↓
@anthropic-ai/claude-code SDK
    ↓
Anthropic API (claude-sonnet-4-5-20250929)
```

### Components

- **ClaudeTextRefiner.swift**: Socket client that communicates with the daemon
- **DaemonManager.swift**: Manages the daemon process lifecycle
- **Protocols.swift**: Defines the `TextRefining` protocol for abstraction
- **chirp-daemon/**: Node.js daemon process using the Claude Code SDK

## User Flow

```
User speaks into microphone
    ↓
Whisper transcribes audio to text
    ↓
Text accumulated in buffer
    ↓
User stops speaking (pushes key again or speaks "park")
    ↓
If refinement enabled:
    ↓
    Send accumulated text to daemon via socket
    ↓
    Daemon constructs prompt with system instructions
    ↓
    Daemon calls Claude via SDK
    ↓
    Claude returns refined text
    ↓
    Daemon sends refined text back to app
    ↓
Refined (or original) text typed into active application
```

### Detailed Sequence Diagram

```
┌──────┐         ┌─────────┐         ┌────────┐         ┌─────────┐         ┌──────────┐
│ User │         │  Chirp  │         │ Daemon │         │ Claude  │         │ Target   │
│      │         │   App   │         │        │         │   SDK   │         │   App    │
└──┬───┘         └────┬────┘         └───┬────┘         └────┬────┘         └────┬─────┘
   │                  │                  │                   │                   │
   │  Press hotkey    │                  │                   │                   │
   ├─────────────────>│                  │                   │                   │
   │                  │                  │                   │                   │
   │  Speak: "Send... │                  │                   │                   │
   ├─────────────────>│                  │                   │                   │
   │  ...an email to" │ Whisper          │                   │                   │
   │                  │ transcribes      │                   │                   │
   │                  │ (accumulates)    │                   │                   │
   │                  │                  │                   │                   │
   │  Press hotkey    │                  │                   │                   │
   │  or say "park"   │                  │                   │                   │
   ├─────────────────>│                  │                   │                   │
   │                  │                  │                   │                   │
   │                  │ refine(text)     │                   │                   │
   │                  ├─────────────────>│                   │                   │
   │                  │                  │                   │                   │
   │                  │                  │ SDK.send(prompt)  │                   │
   │                  │                  ├──────────────────>│                   │
   │                  │                  │                   │                   │
   │                  │                  │                   │ API call          │
   │                  │                  │                   ├──────────────────>│
   │                  │                  │                   │ to Anthropic      │
   │                  │                  │                   │                   │
   │                  │                  │                   │ Refined text      │
   │                  │                  │                   │<──────────────────┤
   │                  │                  │                   │                   │
   │                  │                  │ Refined text      │                   │
   │                  │                  │<──────────────────┤                   │
   │                  │                  │                   │                   │
   │                  │ JSON response    │                   │                   │
   │                  │<─────────────────┤                   │                   │
   │                  │                  │                   │                   │
   │                  │                  │                   │ Type refined text │
   │                  │                  │                   │  ────────────────>│
   │                  │                  │                   │                   │
   │  See typed text  │                  │                   │                   │
   │<─────────────────┼──────────────────┼───────────────────┼───────────────────┤
   │                  │                  │                   │                   │
```

## Configuration

All configuration is stored in `UserDefaults` and accessible through the Chirp settings interface.

### Available Options

| Option | Key | Type | Default | Description |
|--------|-----|------|---------|-------------|
| **Enable Refinement** | `enableClaudeRefinement` | Boolean | `false` | Master toggle for the feature |
| **Model** | `claudeModel` | String | `claude-sonnet-4-5-20250929` | The Claude model to use |
| **System Prompt** | `claudeSystemPrompt` | String | See below | Instructions for Claude |

### Default System Prompt

```
You are refining voice-transcribed text. Fix obvious transcription errors, improve clarity and grammar, but preserve the speaker's intent and tone. Keep it concise.
```

Users can customize this prompt to adjust Claude's behavior (e.g., "make it more formal", "fix grammar only", etc.).

## Daemon Lifecycle

The daemon is managed by `DaemonManager.swift` and follows a robust lifecycle pattern.

### Startup

1. **On app launch** (if refinement is enabled):
   - `DaemonManager.shared.start()` is called
   - Daemon process spawned: `node /path/to/chirp-daemon/dist/index.js`
   - Manager polls for socket file creation at `/tmp/chirp-claude.sock`
   - Polling: 100 checks × 100ms intervals = 10-second timeout
   - If socket appears, daemon is considered ready
   - If timeout expires, startup is considered failed

### Monitoring

- Daemon process is monitored via its `terminationHandler`
- If the process terminates unexpectedly, auto-restart is triggered

### Auto-Restart

Uses exponential backoff to avoid restart storms:

1. First restart: 1-second delay
2. Second restart: 2-second delay
3. Third restart: 4-second delay
4. Fourth restart: 8-second delay
5. Fifth restart: 16-second delay
6. Subsequent restarts: 30-second delay (capped)

After 5 consecutive failures, the daemon is marked as failed and auto-restart is disabled. Manual re-enable required (toggle setting off/on or restart app).

### Shutdown

- **On app quit**: `DaemonManager.shared.stop()` terminates the daemon process
- **On feature disable**: Daemon is stopped and will not auto-restart
- **Manual restart**: Available via settings UI (stop + start)

## Communication Protocol

Communication uses newline-delimited JSON over a Unix domain socket.

### Socket Location

```
/tmp/chirp-claude.sock
```

### Request Format

```json
{
  "action": "refine",
  "text": "send a email to john about the meeting",
  "systemPrompt": "Fix grammar and improve clarity while preserving intent.",
  "model": "claude-sonnet-4-5-20250929"
}
```

Each request must be followed by a newline (`\n`).

### Response Format

**Success:**

```json
{
  "ok": true,
  "refined": "Send an email to John about the meeting."
}
```

**Error:**

```json
{
  "ok": false,
  "error": "API rate limit exceeded. Please try again later."
}
```

Each response is terminated by a newline (`\n`).

### Error Handling

- **Connection failure**: Fallback to unrefined text
- **Timeout** (30 seconds): Fallback to unrefined text
- **Malformed response**: Fallback to unrefined text
- **Daemon not running**: Fallback to unrefined text
- All errors are logged to Console.app under the Chirp subsystem

## Development Setup

### Prerequisites

- **Node.js**: v18 or later
- **pnpm**: Latest version (`npm install -g pnpm`)
- **Anthropic API key**: Configured via Claude Code CLI (`claude config`)

### Building the Daemon

```bash
cd chirp-daemon
pnpm install
pnpm run build
```

This produces `chirp-daemon/dist/index.js`, which is the entry point the Swift app launches.

### Running the Daemon Standalone (for testing)

```bash
cd chirp-daemon
pnpm run dev
```

This starts the daemon with auto-reload on file changes. The socket is created at `/tmp/chirp-claude.sock`.

### Testing the Socket

```bash
# In one terminal, start the daemon
cd chirp-daemon && pnpm run dev

# In another terminal, send a test request
echo '{"action":"refine","text":"send a email to john","systemPrompt":"Fix grammar.","model":"claude-sonnet-4-5-20250929"}' | nc -U /tmp/chirp-claude.sock
```

Expected response:

```json
{"ok":true,"refined":"Send an email to John."}
```

### Dependencies

The daemon relies on:

- **@anthropic-ai/claude-code**: Official Claude Code SDK for Node.js
- **Socket communication**: Node.js built-in `net` module
- **JSON parsing**: Node.js built-in `JSON` module

### API Key Configuration

The `@anthropic-ai/claude-code` SDK reads the API key from the Claude Code configuration:

```bash
# Configure Claude Code (one-time setup)
claude config

# Verify configuration
claude config list
```

The SDK automatically uses the configured API key. No environment variables needed.

## File Map

| File | Purpose |
|------|---------|
| `Sources/Chirp/ClaudeTextRefiner.swift` | Socket client implementation |
| `Sources/Chirp/DaemonManager.swift` | Process lifecycle management |
| `Sources/Chirp/Protocols.swift` | `TextRefining` protocol definition |
| `chirp-daemon/src/index.ts` | Node.js daemon server implementation |
| `chirp-daemon/package.json` | Daemon dependencies and scripts |
| `chirp-daemon/tsconfig.json` | TypeScript compiler configuration |
| `/tmp/chirp-claude.sock` | Unix domain socket (runtime) |

## Troubleshooting

### Daemon won't start

1. Check if Node.js is installed: `node --version`
2. Check if daemon is built: `ls chirp-daemon/dist/index.js`
3. Check Console.app for Chirp logs (filter by process "Chirp")
4. Try manual start: `node chirp-daemon/dist/index.js`

### Socket connection fails

1. Verify socket exists: `ls -l /tmp/chirp-claude.sock`
2. Check socket permissions: Should be user-readable/writable
3. Try manual socket test (see "Testing the Socket" above)
4. Check if another process is using the socket: `lsof /tmp/chirp-claude.sock`

### API calls fail

1. Verify Claude Code is configured: `claude config list`
2. Check API key validity via Claude Code CLI
3. Check network connectivity
4. Review error messages in Console.app

### Refinement produces unexpected results

1. Review system prompt in settings
2. Try a more specific prompt (e.g., "Only fix grammar, do not rephrase")
3. Test with different models if available
4. Check Claude's response in daemon logs (if logging enabled)

## Security Considerations

- **Socket permissions**: The Unix socket is created with user-only permissions (0600)
- **API key storage**: API key is stored securely by Claude Code CLI, not in Chirp
- **Process isolation**: Daemon runs as a separate process with its own memory space
- **No data retention**: Neither the daemon nor Chirp stores transcriptions or refined text

## Performance

- **Typical latency**: 1-3 seconds for short texts (< 100 words)
- **Longer texts**: 3-5 seconds for 100-300 words
- **Timeout**: 30 seconds (configurable in `ClaudeTextRefiner.swift`)
- **Concurrent requests**: Not supported (single socket, sequential processing)

## Future Enhancements

Potential improvements:

- [ ] Multiple model support in UI
- [ ] Custom prompt templates (saved presets)
- [ ] Refinement history/undo
- [ ] Streaming responses for longer texts
- [ ] Offline mode with local LLM fallback
- [ ] Per-app refinement profiles
- [ ] Confidence indicators for refinements
