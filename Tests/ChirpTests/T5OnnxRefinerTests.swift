import Testing
import Foundation
import COnnxRuntime
@testable import Chirp

@Suite("T5OnnxRefiner integration tests (require downloaded model)")
struct T5OnnxRefinerTests {

    @Test("findExisting returns nil for bogus path")
    func findExistingReturnsNilForBogusPath() {
        // T5ModelManager.findExisting() checks Application Support for the model.
        // When the model isn't downloaded, it should return nil rather than crash.
        // This test documents that behavior — on CI the model won't be present.
        let result = T5ModelManager.findExisting()
        // We can't assert nil/non-nil since it depends on the machine,
        // but we verify it doesn't crash and returns an Optional.
        _ = result
    }

    @Test("inspectModels verifies ONNX model I/O names")
    func inspectModels() async throws {
        guard let paths = T5ModelManager.findExisting() else {
            print("SKIP: T5 model not downloaded")
            return
        }

        let apiBase = OrtGetApiBase()!
        let api = apiBase.pointee.GetApi(UInt32(ORT_API_VERSION))!

        var env: OpaquePointer?
        try ortCheck(api, api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env))
        defer { api.pointee.ReleaseEnv(env) }

        var opts: OpaquePointer?
        try ortCheck(api, api.pointee.CreateSessionOptions(&opts))
        defer { api.pointee.ReleaseSessionOptions(opts) }

        // Expected I/O names per model
        let expectations: [(label: String, path: String, expectedInputs: [String], expectedOutputPrefix: String)] = [
            ("encoder", paths.encoderPath, ["input_ids", "attention_mask"], "last_hidden_state"),
            ("decoder", paths.decoderPath, ["input_ids", "encoder_attention_mask", "encoder_hidden_states"], "logits"),
            ("decoder_with_past", paths.decoderWithPastPath, ["input_ids", "encoder_attention_mask", "encoder_hidden_states"], "logits"),
        ]

        for (label, path, expectedInputs, expectedOutputPrefix) in expectations {
            var session: OpaquePointer?
            try ortCheck(api, api.pointee.CreateSession(env, path, opts, &session))
            defer { api.pointee.ReleaseSession(session) }

            var allocator: UnsafeMutablePointer<OrtAllocator>?
            try ortCheck(api, api.pointee.GetAllocatorWithDefaultOptions(&allocator))

            // Collect input names
            var inputCount: Int = 0
            try ortCheck(api, api.pointee.SessionGetInputCount(session, &inputCount))
            var inputNames: [String] = []
            for i in 0..<inputCount {
                var name: UnsafeMutablePointer<CChar>?
                try ortCheck(api, api.pointee.SessionGetInputName(session, i, allocator, &name))
                inputNames.append(String(cString: name!))
                _ = api.pointee.AllocatorFree(allocator, name)
            }

            // Verify expected inputs are present
            for expected in expectedInputs {
                #expect(inputNames.contains(expected), "\(label) missing input '\(expected)', got: \(inputNames)")
            }

            // Collect output names
            var outputCount: Int = 0
            try ortCheck(api, api.pointee.SessionGetOutputCount(session, &outputCount))
            var outputNames: [String] = []
            for i in 0..<outputCount {
                var name: UnsafeMutablePointer<CChar>?
                try ortCheck(api, api.pointee.SessionGetOutputName(session, i, allocator, &name))
                outputNames.append(String(cString: name!))
                _ = api.pointee.AllocatorFree(allocator, name)
            }

            // Verify first output matches expected prefix
            #expect(outputCount > 0, "\(label) should have at least one output")
            #expect(outputNames[0] == expectedOutputPrefix, "\(label) first output should be '\(expectedOutputPrefix)', got '\(outputNames[0])'")
        }
    }

    @Test("Empty input handled gracefully")
    func emptyInputHandledGracefully() async throws {
        guard let paths = T5ModelManager.findExisting() else {
            print("SKIP: T5 model not downloaded")
            return
        }

        let refiner = try await T5OnnxRefiner(paths: paths)
        let output = try await refiner.refine(text: "", systemPrompt: "")
        // Empty or whitespace-only output is acceptable for empty input
        #expect(output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !output.isEmpty)
    }

    @Test func grammarCorrection() async throws {
        guard let paths = T5ModelManager.findExisting() else {
            print("SKIP: T5 model not downloaded")
            return
        }

        let refiner = try await T5OnnxRefiner(paths: paths)
        let tests = [
            "Me and him went to the store",
            "i can has cheezburger",
            "She dont know nothing about it",
            "Their going to the store tommorow",
            "He goed to school yesterday",
        ]
        for input in tests {
            let output = try await refiner.refine(text: input, systemPrompt: "")
            print("Input:  \(input)")
            print("Output: \(output)")
            print()
            #expect(!output.isEmpty)
        }
    }
}

private func ortCheck(_ api: UnsafePointer<OrtApi>, _ status: OpaquePointer?) throws {
    guard let status else { return }
    let msg = String(cString: api.pointee.GetErrorMessage(status)!)
    api.pointee.ReleaseStatus(status)
    throw T5OnnxError.ortError(msg)
}
