// ClaudeTextRefiner.swift — Sends transcribed text to a local Claude
// sidecar process over a Unix socket for grammar/punctuation cleanup.
// Conforms to TextRefining protocol.

import Foundation

public struct ClaudeRefineConfig: Sendable {
    public var prompt: String {
        get {
            UserDefaults.standard.string(forKey: "claudeRefinePrompt")
                ?? """
                You are a text refinement tool for a voice dictation app called Chirp. Your ONLY role is to clean up transcribed speech.

                Rules:
                - Fix grammar, punctuation, and capitalization
                - Remove filler words (um, uh, like, you know) and verbal artifacts
                - Preserve the original meaning, tone, and intent exactly
                - Output ONLY the cleaned text — no greetings, explanations, commentary, or markdown
                - Do NOT act as a conversational assistant, coding helper, or chatbot
                - If the input is a question, statement, or any other content, clean it up and return it as-is — do NOT answer or respond to it
                """
        }
        set { UserDefaults.standard.set(newValue, forKey: "claudeRefinePrompt") }
    }

    public var model: String {
        get {
            UserDefaults.standard.string(forKey: "claudeRefineModel") ?? "claude-sonnet-4-6-latest"
        }
        set { UserDefaults.standard.set(newValue, forKey: "claudeRefineModel") }
    }
}

@MainActor
final class ClaudeTextRefiner: TextRefining {
    private let socketPath = NSTemporaryDirectory() + "chirp-claude.sock"
    private let timeout: TimeInterval = 30

    /// Sends transcribed text to the Claude daemon for grammar/punctuation cleanup.
    /// - Parameters:
    ///   - text: Raw transcribed text from the speech recognizer.
    ///   - systemPrompt: System prompt instructing Claude how to refine the text.
    ///   - model: Claude model ID (default: claude-sonnet-4-6-latest).
    /// - Returns: Refined text with corrected grammar, punctuation, and removed filler words.
    /// - Throws: `RefineError` if socket connection, communication, or parsing fails.
    func refine(text: String, systemPrompt: String, model: String = "claude-sonnet-4-6-latest") async throws -> String {
        let request: [String: Any] = [
            "action": "refine",
            "model": model,
            "systemPrompt": systemPrompt,
            "text": text,
        ]
        let requestData = try JSONSerialization.data(withJSONObject: request)

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.sendSocketRequest(requestData)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Refines text using the user's configured model preference from `ClaudeRefineConfig`.
    /// - Parameters:
    ///   - text: Raw transcribed text.
    ///   - systemPrompt: System prompt instructing Claude how to refine the text.
    /// - Returns: Refined text.
    /// - Throws: `RefineError` if refinement fails.
    func refine(text: String, systemPrompt: String) async throws -> String {
        try await refine(text: text, systemPrompt: systemPrompt, model: ClaudeRefineConfig().model)
    }

    // MARK: - Socket I/O

    private nonisolated func sendSocketRequest(_ requestData: Data) throws -> String {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw RefineError.socketCreationFailed(errno: errno)
        }
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            throw RefineError.pathTooLong
        }
        // Copy socket path bytes into the fixed-size sun_path tuple.
        // `sun_path` is a tuple of Int8, not a CChar array, so we use
        // withUnsafeMutablePointer + withMemoryRebound to write into it.
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for i in 0..<pathBytes.count {
                    dest[i] = pathBytes[i]
                }
            }
        }

        let addrSize = socklen_t(MemoryLayout<sockaddr_un>.size)
        let connectResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                connect(fd, sockPtr, addrSize)
            }
        }
        guard connectResult == 0 else {
            throw RefineError.connectionFailed(errno: errno)
        }

        // Set receive timeout
        var tv = timeval(tv_sec: Int(timeout), tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        // Send newline-terminated JSON. The daemon protocol uses newline as the message delimiter.
        var payload = requestData
        payload.append(0x0A) // newline
        let written = payload.withUnsafeBytes { buf in
            send(fd, buf.baseAddress!, buf.count, 0)
        }
        guard written == payload.count else {
            throw RefineError.sendFailed(errno: errno)
        }

        // Read response until newline delimiter.
        // The daemon sends newline-terminated JSON. We accumulate data
        // in a loop until we see the newline byte (0x0A).
        var responseData = Data()
        let bufferSize = 4096
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }

        let maxResponseSize = 1_048_576 // 1MB
        let deadline = DispatchTime.now().uptimeNanoseconds + UInt64(timeout * 1_000_000_000)

        while true {
            if responseData.count > maxResponseSize {
                throw RefineError.responseTooLarge
            }
            if DispatchTime.now().uptimeNanoseconds > deadline {
                throw RefineError.timeout
            }
            let bytesRead = recv(fd, buffer, bufferSize, 0)
            if bytesRead <= 0 {
                throw RefineError.readFailed(errno: errno)
            }
            responseData.append(buffer, count: bytesRead)
            if responseData.contains(0x0A) { break }
        }

        // Trim trailing newline and parse
        if responseData.last == 0x0A {
            responseData.removeLast()
        }

        guard let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any] else {
            throw RefineError.invalidResponse
        }

        if let error = json["error"] as? String {
            throw RefineError.serverError(error)
        }

        guard let result = json["refined"] as? String else {
            throw RefineError.invalidResponse
        }

        return result
    }
}

enum RefineError: LocalizedError {
    case socketCreationFailed(errno: Int32)
    case pathTooLong
    case connectionFailed(errno: Int32)
    case sendFailed(errno: Int32)
    case readFailed(errno: Int32)
    case invalidResponse
    case serverError(String)
    case responseTooLarge
    case timeout

    var errorDescription: String? {
        switch self {
        case .socketCreationFailed(let e): "Failed to create socket: \(String(cString: strerror(e)))"
        case .pathTooLong: "Socket path exceeds maximum length"
        case .connectionFailed(let e): "Failed to connect to Claude sidecar: \(String(cString: strerror(e)))"
        case .sendFailed(let e): "Failed to send request: \(String(cString: strerror(e)))"
        case .readFailed(let e): "Failed to read response: \(String(cString: strerror(e)))"
        case .invalidResponse: "Invalid response from Claude sidecar"
        case .serverError(let msg): "Claude sidecar error: \(msg)"
        case .responseTooLarge: "Response exceeded maximum size"
        case .timeout: "Request timed out"
        }
    }
}
