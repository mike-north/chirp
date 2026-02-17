// Protocols.swift — Dependency injection boundaries.
// AppState depends on these protocols rather than concrete types,
// enabling mock-based testing without audio hardware or ML models.

import Foundation

struct ModelPaths: Sendable {
    let modelDir: String
    let vadPath: String
    let variant: ModelVariant
}

protocol TranscriberProtocol: Sendable {
    func initialize(paths: ModelPaths) async -> Bool
    func feedAudio(samples: [Float]) async -> [String]
    func peekTranscription() async -> String?
    func flush() async -> String
    func resetVAD() async
}

@MainActor protocol AudioRecording {
    func requestMicrophoneAccess() async -> Bool
    func prepare()
    func startRecording(onSamples: @escaping @Sendable ([Float]) -> Void)
    func stopRecording()
}

extension AudioRecording {
    func prepare() {}
    func requestMicrophoneAccess() async -> Bool { true }
}

@MainActor protocol TextInserting {
    func checkAccessibilityPermission()
    func typeText(_ text: String)
    func deleteBackward(count: Int)
}

/// Abstraction for post-transcription text refinement (e.g., grammar, punctuation, filler word removal).
/// Implementations send transcribed text to an LLM or other service for cleanup before insertion.
@MainActor public protocol TextRefining {
    /// Refines raw transcribed text using the provided system prompt.
    /// - Parameters:
    ///   - text: Raw transcription output.
    ///   - systemPrompt: Instructions for how to refine the text.
    /// - Returns: Cleaned-up text with corrected grammar, punctuation, etc.
    /// - Throws: If refinement fails (network, parsing, or service errors).
    func refine(text: String, systemPrompt: String) async throws -> String
}
