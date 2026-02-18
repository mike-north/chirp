import Foundation
import COnnxRuntime
import Tokenizers

// MARK: - ORT Helper

/// Thin wrapper that checks ORT status codes and throws on failure.
private func ortCheck(_ api: UnsafePointer<OrtApi>, _ status: OpaquePointer?) throws {
    guard let status else { return } // nil means success
    let msg = String(cString: api.pointee.GetErrorMessage(status)!)
    api.pointee.ReleaseStatus(status)
    throw T5OnnxError.ortError(msg)
}

enum T5OnnxError: LocalizedError {
    case ortError(String)
    case tokenizationFailed
    case encoderOutputMissing
    case decoderOutputMissing
    case maxLengthExceeded
    case configMissing(String)
    case sessionCreationFailed(String)

    var errorDescription: String? {
        switch self {
        case .ortError(let msg): "ORT error: \(msg)"
        case .tokenizationFailed: "Failed to tokenize input"
        case .encoderOutputMissing: "Encoder did not produce hidden states"
        case .decoderOutputMissing: "Decoder did not produce logits"
        case .maxLengthExceeded: "Decoding exceeded maximum length"
        case .configMissing(let msg): "Model config error: \(msg)"
        case .sessionCreationFailed(let name): "Failed to create ORT session: \(name)"
        }
    }
}

// MARK: - T5OnnxRefiner

/// Uses split decoder models: `decoder_model` for the first step (no past KV cache)
/// and `decoder_with_past_model` for subsequent steps (with KV cache).
/// This avoids ORT If-node shape validation issues with the merged decoder model.
@MainActor
final class T5OnnxRefiner: TextRefining {
    private let paths: T5ModelPaths
    nonisolated(unsafe) private let tokenizer: Tokenizer
    nonisolated(unsafe) private let api: UnsafePointer<OrtApi>
    nonisolated(unsafe) private let env: OpaquePointer
    nonisolated(unsafe) private let encoderSession: OpaquePointer
    nonisolated(unsafe) private let decoderSession: OpaquePointer
    nonisolated(unsafe) private let decoderWithPastSession: OpaquePointer
    private let maxLength: Int
    private let numLayers: Int

    init(paths: T5ModelPaths, maxLength: Int = 512) async throws {
        self.paths = paths
        self.maxLength = maxLength

        let tokenizerURL = URL(fileURLWithPath: paths.tokenizerDir, isDirectory: true)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)

        // S4/M1: Throwing config parsing
        self.numLayers = try Self.parseNumDecoderLayers(from: tokenizerURL)

        let apiBase = OrtGetApiBase()!
        let api = apiBase.pointee.GetApi(UInt32(ORT_API_VERSION))!
        self.api = api

        // H2: Replace force unwraps with guard let + throw
        var envPtr: OpaquePointer?
        try ortCheck(api, api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "chirp-t5", &envPtr))
        guard let envVal = envPtr else {
            throw T5OnnxError.sessionCreationFailed("ORT environment")
        }
        self.env = envVal

        var opts: OpaquePointer?
        try ortCheck(api, api.pointee.CreateSessionOptions(&opts))
        try ortCheck(api, api.pointee.SetIntraOpNumThreads(opts, 2))
        try ortCheck(api, api.pointee.SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL))
        defer { api.pointee.ReleaseSessionOptions(opts) }

        var encSession: OpaquePointer?
        try ortCheck(api, api.pointee.CreateSession(envPtr, paths.encoderPath, opts, &encSession))
        guard let encSessionVal = encSession else {
            throw T5OnnxError.sessionCreationFailed("encoder")
        }
        self.encoderSession = encSessionVal

        var decSession: OpaquePointer?
        try ortCheck(api, api.pointee.CreateSession(envPtr, paths.decoderPath, opts, &decSession))
        guard let decSessionVal = decSession else {
            throw T5OnnxError.sessionCreationFailed("decoder")
        }
        self.decoderSession = decSessionVal

        var decPastSession: OpaquePointer?
        try ortCheck(api, api.pointee.CreateSession(envPtr, paths.decoderWithPastPath, opts, &decPastSession))
        guard let decPastSessionVal = decPastSession else {
            throw T5OnnxError.sessionCreationFailed("decoder_with_past")
        }
        self.decoderWithPastSession = decPastSessionVal
    }

    deinit {
        api.pointee.ReleaseSession(encoderSession)
        api.pointee.ReleaseSession(decoderSession)
        api.pointee.ReleaseSession(decoderWithPastSession)
        api.pointee.ReleaseEnv(env)
    }

    // S4/M1: Throwing helper for config.json parsing
    private static func parseNumDecoderLayers(from modelFolder: URL) throws -> Int {
        let configURL = modelFolder.appendingPathComponent("config.json")
        let configData: Data
        do {
            configData = try Data(contentsOf: configURL)
        } catch {
            throw T5OnnxError.configMissing("Cannot read config.json: \(error.localizedDescription)")
        }

        guard let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw T5OnnxError.configMissing("config.json is not a valid JSON dictionary")
        }

        guard let layers = config["num_decoder_layers"] as? Int else {
            throw T5OnnxError.configMissing("num_decoder_layers not found or not an integer in config.json")
        }

        return layers
    }

    func refine(text: String, systemPrompt: String) async throws -> String {
        let prefixedText: String
        if systemPrompt.isEmpty {
            prefixedText = text
        } else {
            prefixedText = systemPrompt.trimmingCharacters(in: .whitespaces) + ": " + text
        }
        let encoded = tokenizer.encode(text: prefixedText)
        let inputIds = encoded.map { Int64($0) }
        let attentionMask = [Int64](repeating: 1, count: inputIds.count)

        let outputTokens = try await runInference(
            inputIds: inputIds, attentionMask: attentionMask
        )

        let outputIds = outputTokens.map { Int($0) }
        return tokenizer.decode(tokens: outputIds)
    }

    nonisolated private func runInference(
        inputIds: sending [Int64], attentionMask: sending [Int64]
    ) async throws -> sending [Int64] {
        let api = self.api
        let encoderSession = self.encoderSession
        let decoderSession = self.decoderSession
        let decoderWithPastSession = self.decoderWithPastSession
        let maxLen = self.maxLength
        let layers = self.numLayers
        let seqLen = inputIds.count

        var memInfo: OpaquePointer?
        try ortCheck(api, api.pointee.CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &memInfo
        ))
        defer { api.pointee.ReleaseMemoryInfo(memInfo) }

        // 1. Run encoder
        let hiddenStates = try inputIds.withUnsafeBufferPointer { idsBuf in
            try attentionMask.withUnsafeBufferPointer { maskBuf in
                try Self.runEncoder(
                    api: api, session: encoderSession, memInfo: memInfo!,
                    inputIds: idsBuf, attentionMask: maskBuf, seqLen: seqLen
                )
            }
        }

        let eosToken: Int64 = 1
        let padToken: Int64 = 0
        var decoderInputIds: [Int64] = [padToken]
        var tokens: [Int64] = []

        // 2. First decoder step (no past KV cache)
        let (firstToken, kvCache) = try decoderInputIds.withUnsafeBufferPointer { decIdsBuf in
            try attentionMask.withUnsafeBufferPointer { maskBuf in
                try Self.runFirstDecoderStep(
                    api: api, session: decoderSession, memInfo: memInfo!,
                    decoderInputIds: decIdsBuf,
                    encoderAttentionMask: maskBuf,
                    encoderHiddenStates: hiddenStates,
                    seqLen: seqLen,
                    numLayers: layers
                )
            }
        }

        if firstToken == eosToken { return tokens }
        tokens.append(firstToken)
        decoderInputIds = [firstToken]

        // 3. Subsequent decoder steps (with past KV cache)
        var currentKVCache = kvCache
        for _ in 1..<maxLen {
            let (nextToken, newKVCache) = try decoderInputIds.withUnsafeBufferPointer { decIdsBuf in
                try attentionMask.withUnsafeBufferPointer { maskBuf in
                    try Self.runDecoderWithPastStep(
                        api: api, session: decoderWithPastSession, memInfo: memInfo!,
                        decoderInputIds: decIdsBuf,
                        encoderAttentionMask: maskBuf,
                        encoderHiddenStates: hiddenStates,
                        kvCache: currentKVCache,
                        seqLen: seqLen,
                        numLayers: layers
                    )
                }
            }

            if nextToken == eosToken { break }
            tokens.append(nextToken)
            decoderInputIds = [nextToken]
            currentKVCache = newKVCache
        }

        return tokens
    }

    // MARK: - Encoder

    nonisolated private static func runEncoder(
        api: UnsafePointer<OrtApi>,
        session: OpaquePointer,
        memInfo: OpaquePointer,
        inputIds: UnsafeBufferPointer<Int64>,
        attentionMask: UnsafeBufferPointer<Int64>,
        seqLen: Int
    ) throws -> EncoderOutput {
        let shape: [Int64] = [1, Int64(seqLen)]

        var idsTensor: OpaquePointer?
        try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
            memInfo,
            UnsafeMutableRawPointer(mutating: inputIds.baseAddress!),
            seqLen * MemoryLayout<Int64>.size,
            shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &idsTensor
        ))
        defer { api.pointee.ReleaseValue(idsTensor) }

        var maskTensor: OpaquePointer?
        try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
            memInfo,
            UnsafeMutableRawPointer(mutating: attentionMask.baseAddress!),
            seqLen * MemoryLayout<Int64>.size,
            shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &maskTensor
        ))
        defer { api.pointee.ReleaseValue(maskTensor) }

        let inputNames = ["input_ids", "attention_mask"]
        let outputNames = ["last_hidden_state"]

        var output: OpaquePointer?
        try inputNames.withCArrayOfCStrings { inNames in
            try outputNames.withCArrayOfCStrings { outNames in
                var inputs: [OpaquePointer?] = [idsTensor, maskTensor]
                try ortCheck(api, api.pointee.Run(
                    session, nil,
                    inNames, &inputs, 2,
                    outNames, 1, &output
                ))
            }
        }
        guard let output else { throw T5OnnxError.encoderOutputMissing }

        let result = try extractTensorData(api: api, tensor: output)
        api.pointee.ReleaseValue(output)
        return EncoderOutput(data: result.data, shape: result.shape)
    }

    // MARK: - First Decoder Step (no past KV cache)

    /// Runs decoder_model.onnx which takes: input_ids, encoder_attention_mask, encoder_hidden_states
    /// and produces: logits + present KV cache for all layers.
    nonisolated private static func runFirstDecoderStep(
        api: UnsafePointer<OrtApi>,
        session: OpaquePointer,
        memInfo: OpaquePointer,
        decoderInputIds: UnsafeBufferPointer<Int64>,
        encoderAttentionMask: UnsafeBufferPointer<Int64>,
        encoderHiddenStates: EncoderOutput,
        seqLen: Int,
        numLayers: Int
    ) throws -> (Int64, KVCache) {
        let decSeqLen = decoderInputIds.count

        // C1: Use defer for tensor cleanup
        var decIdsTensor: OpaquePointer?
        let decShape: [Int64] = [1, Int64(decSeqLen)]
        try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
            memInfo,
            UnsafeMutableRawPointer(mutating: decoderInputIds.baseAddress!),
            decSeqLen * MemoryLayout<Int64>.size,
            decShape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &decIdsTensor
        ))
        defer { api.pointee.ReleaseValue(decIdsTensor) }

        var encMaskTensor: OpaquePointer?
        let encMaskShape: [Int64] = [1, Int64(seqLen)]
        try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
            memInfo,
            UnsafeMutableRawPointer(mutating: encoderAttentionMask.baseAddress!),
            seqLen * MemoryLayout<Int64>.size,
            encMaskShape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &encMaskTensor
        ))
        defer { api.pointee.ReleaseValue(encMaskTensor) }

        var hiddenDataCopy = encoderHiddenStates.data
        let hiddenShape = encoderHiddenStates.shape.map { Int64($0) }

        let result: (Int64, KVCache) = try hiddenDataCopy.withUnsafeMutableBufferPointer { hiddenBuf in
            var hiddenTensor: OpaquePointer?
            try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
                memInfo,
                hiddenBuf.baseAddress!,
                hiddenBuf.count * MemoryLayout<Float>.size,
                hiddenShape, hiddenShape.count,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &hiddenTensor
            ))
            defer { api.pointee.ReleaseValue(hiddenTensor) }

            let inputNames = ["encoder_attention_mask", "input_ids", "encoder_hidden_states"]
            var inputTensors: [OpaquePointer?] = [encMaskTensor, decIdsTensor, hiddenTensor]

            var outputNames: [String] = ["logits"]
            for i in 0..<numLayers {
                outputNames.append("present.\(i).decoder.key")
                outputNames.append("present.\(i).decoder.value")
                outputNames.append("present.\(i).encoder.key")
                outputNames.append("present.\(i).encoder.value")
            }

            let numOutputs = outputNames.count
            var outputs = [OpaquePointer?](repeating: nil, count: numOutputs)

            try inputNames.withCArrayOfCStrings { inNames in
                try outputNames.withCArrayOfCStrings { outNames in
                    try ortCheck(api, api.pointee.Run(
                        session, nil,
                        inNames, &inputTensors, inputTensors.count,
                        outNames, numOutputs, &outputs
                    ))
                }
            }

            return try processDecoderOutputs(api: api, outputs: &outputs, decSeqLen: decSeqLen, numLayers: numLayers)
        }

        return result
    }

    // MARK: - Decoder With Past Step (with KV cache)

    /// Runs decoder_with_past_model.onnx which takes: input_ids, encoder_attention_mask,
    /// encoder_hidden_states, and past_key_values for all layers.
    nonisolated private static func runDecoderWithPastStep(
        api: UnsafePointer<OrtApi>,
        session: OpaquePointer,
        memInfo: OpaquePointer,
        decoderInputIds: UnsafeBufferPointer<Int64>,
        encoderAttentionMask: UnsafeBufferPointer<Int64>,
        encoderHiddenStates: EncoderOutput,
        kvCache: KVCache,
        seqLen: Int,
        numLayers: Int
    ) throws -> (Int64, KVCache) {
        let decSeqLen = decoderInputIds.count

        // C1: Use defer for tensor cleanup
        var decIdsTensor: OpaquePointer?
        let decShape: [Int64] = [1, Int64(decSeqLen)]
        try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
            memInfo,
            UnsafeMutableRawPointer(mutating: decoderInputIds.baseAddress!),
            decSeqLen * MemoryLayout<Int64>.size,
            decShape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &decIdsTensor
        ))
        defer { api.pointee.ReleaseValue(decIdsTensor) }

        var encMaskTensor: OpaquePointer?
        let encMaskShape: [Int64] = [1, Int64(seqLen)]
        try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
            memInfo,
            UnsafeMutableRawPointer(mutating: encoderAttentionMask.baseAddress!),
            seqLen * MemoryLayout<Int64>.size,
            encMaskShape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &encMaskTensor
        ))
        defer { api.pointee.ReleaseValue(encMaskTensor) }

        // Allocate stable buffers for KV cache and hidden states
        var kvAllocations: [UnsafeMutableBufferPointer<Float>] = []
        defer { for buf in kvAllocations { buf.deallocate() } }

        for entry in kvCache.entries {
            for tensorData in [entry.decoderKey, entry.decoderValue, entry.encoderKey, entry.encoderValue] {
                let buf = UnsafeMutableBufferPointer<Float>.allocate(capacity: max(1, tensorData.data.count))
                // C2: Initialize buffer even when data is empty
                if tensorData.data.isEmpty {
                    buf.initialize(repeating: 0)
                } else {
                    _ = buf.initialize(from: tensorData.data)
                }
                kvAllocations.append(buf)
            }
        }

        // M2: Pre-compute KV input name arrays before the loop
        let kvTypes = ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]
        var kvInputNames: [String] = []
        for idx in 0..<kvCache.entries.count {
            for kvType in kvTypes {
                kvInputNames.append("past_key_values.\(idx).\(kvType)")
            }
        }

        var hiddenDataCopy = encoderHiddenStates.data
        let hiddenShape = encoderHiddenStates.shape.map { Int64($0) }

        let result: (Int64, KVCache) = try hiddenDataCopy.withUnsafeMutableBufferPointer { hiddenBuf in
            var hiddenTensor: OpaquePointer?
            try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
                memInfo,
                hiddenBuf.baseAddress!,
                hiddenBuf.count * MemoryLayout<Float>.size,
                hiddenShape, hiddenShape.count,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &hiddenTensor
            ))
            defer { api.pointee.ReleaseValue(hiddenTensor) }

            var inputNames: [String] = [
                "input_ids",
                "encoder_attention_mask",
                "encoder_hidden_states",
            ]
            var inputTensors: [OpaquePointer?] = [decIdsTensor, encMaskTensor, hiddenTensor]

            // Add past KV cache inputs
            var kvTensors: [OpaquePointer?] = []
            for (idx, entry) in kvCache.entries.enumerated() {
                for (j, tensorData) in [entry.decoderKey, entry.decoderValue, entry.encoderKey, entry.encoderValue].enumerated() {
                    let bufIdx = idx * 4 + j
                    let shape = tensorData.shape.map { Int64($0) }
                    var tensor: OpaquePointer?
                    try ortCheck(api, api.pointee.CreateTensorWithDataAsOrtValue(
                        memInfo,
                        kvAllocations[bufIdx].baseAddress!,
                        tensorData.data.count * MemoryLayout<Float>.size,
                        shape, shape.count,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                        &tensor
                    ))
                    kvTensors.append(tensor)
                    inputNames.append(kvInputNames[idx * 4 + j])
                }
            }
            // C1: Defer cleanup for all KV tensors
            defer { for t in kvTensors { api.pointee.ReleaseValue(t) } }
            inputTensors.append(contentsOf: kvTensors)

            // decoder_with_past only outputs decoder KV (encoder KV doesn't change)
            var outputNames: [String] = ["logits"]
            for i in 0..<numLayers {
                outputNames.append("present.\(i).decoder.key")
                outputNames.append("present.\(i).decoder.value")
            }

            let numOutputs = outputNames.count
            var outputs = [OpaquePointer?](repeating: nil, count: numOutputs)

            try inputNames.withCArrayOfCStrings { inNames in
                try outputNames.withCArrayOfCStrings { outNames in
                    try ortCheck(api, api.pointee.Run(
                        session, nil,
                        inNames, &inputTensors, inputTensors.count,
                        outNames, numOutputs, &outputs
                    ))
                }
            }

            return try processDecoderWithPastOutputs(api: api, outputs: &outputs, decSeqLen: decSeqLen, previousKVCache: kvCache, numLayers: numLayers)
        }

        return result
    }

    // MARK: - Output Processing

    /// Process outputs from the first decoder step (includes encoder KV cache)
    nonisolated private static func processDecoderOutputs(
        api: UnsafePointer<OrtApi>,
        outputs: inout [OpaquePointer?],
        decSeqLen: Int,
        numLayers: Int
    ) throws -> (Int64, KVCache) {
        guard let logitsOutput = outputs[0] else { throw T5OnnxError.decoderOutputMissing }
        var logitsData: UnsafeMutableRawPointer?
        try ortCheck(api, api.pointee.GetTensorMutableData(logitsOutput, &logitsData))

        var logitsShapeInfo: OpaquePointer?
        try ortCheck(api, api.pointee.GetTensorTypeAndShape(logitsOutput, &logitsShapeInfo))
        var logitsDimCount: Int = 0
        try ortCheck(api, api.pointee.GetDimensionsCount(logitsShapeInfo, &logitsDimCount))
        var logitsDims = [Int64](repeating: 0, count: logitsDimCount)
        try ortCheck(api, api.pointee.GetDimensions(logitsShapeInfo, &logitsDims, logitsDimCount))
        api.pointee.ReleaseTensorTypeAndShapeInfo(logitsShapeInfo)

        let vocabSize = Int(logitsDims[logitsDimCount - 1])
        // H3: Guard against empty vocab
        guard vocabSize > 0 else { throw T5OnnxError.decoderOutputMissing }
        guard let logitsData else { throw T5OnnxError.decoderOutputMissing }
        let floatPtr = logitsData.assumingMemoryBound(to: Float.self)
        let lastTokenOffset = (decSeqLen - 1) * vocabSize
        let logitsSlice = UnsafeBufferPointer(start: floatPtr + lastTokenOffset, count: vocabSize)

        // Greedy: argmax
        var maxIdx: Int = 0
        var maxVal: Float = logitsSlice[0]
        for i in 1..<vocabSize {
            if logitsSlice[i] > maxVal {
                maxVal = logitsSlice[i]
                maxIdx = i
            }
        }
        let nextToken = Int64(maxIdx)

        // Extract new KV cache from outputs
        var newEntries: [KVCacheEntry] = []
        for i in 0..<numLayers {
            let baseIdx = 1 + i * 4
            let dk = try extractTensorData(api: api, tensor: outputs[baseIdx]!)
            let dv = try extractTensorData(api: api, tensor: outputs[baseIdx + 1]!)
            let ek = try extractTensorData(api: api, tensor: outputs[baseIdx + 2]!)
            let ev = try extractTensorData(api: api, tensor: outputs[baseIdx + 3]!)
            newEntries.append(KVCacheEntry(decoderKey: dk, decoderValue: dv, encoderKey: ek, encoderValue: ev))
        }

        for o in outputs { api.pointee.ReleaseValue(o) }

        return (nextToken, KVCache(entries: newEntries))
    }

    /// Process outputs from decoder_with_past (only decoder KV; encoder KV passed through)
    nonisolated private static func processDecoderWithPastOutputs(
        api: UnsafePointer<OrtApi>,
        outputs: inout [OpaquePointer?],
        decSeqLen: Int,
        previousKVCache: KVCache,
        numLayers: Int
    ) throws -> (Int64, KVCache) {
        guard let logitsOutput = outputs[0] else { throw T5OnnxError.decoderOutputMissing }
        var logitsData: UnsafeMutableRawPointer?
        try ortCheck(api, api.pointee.GetTensorMutableData(logitsOutput, &logitsData))

        var logitsShapeInfo: OpaquePointer?
        try ortCheck(api, api.pointee.GetTensorTypeAndShape(logitsOutput, &logitsShapeInfo))
        var logitsDimCount: Int = 0
        try ortCheck(api, api.pointee.GetDimensionsCount(logitsShapeInfo, &logitsDimCount))
        var logitsDims = [Int64](repeating: 0, count: logitsDimCount)
        try ortCheck(api, api.pointee.GetDimensions(logitsShapeInfo, &logitsDims, logitsDimCount))
        api.pointee.ReleaseTensorTypeAndShapeInfo(logitsShapeInfo)

        let vocabSize = Int(logitsDims[logitsDimCount - 1])
        // H3: Guard against empty vocab
        guard vocabSize > 0 else { throw T5OnnxError.decoderOutputMissing }
        guard let logitsData else { throw T5OnnxError.decoderOutputMissing }
        let floatPtr = logitsData.assumingMemoryBound(to: Float.self)
        let lastTokenOffset = (decSeqLen - 1) * vocabSize
        let logitsSlice = UnsafeBufferPointer(start: floatPtr + lastTokenOffset, count: vocabSize)

        var maxIdx: Int = 0
        var maxVal: Float = logitsSlice[0]
        for i in 1..<vocabSize {
            if logitsSlice[i] > maxVal {
                maxVal = logitsSlice[i]
                maxIdx = i
            }
        }
        let nextToken = Int64(maxIdx)

        // Outputs: logits, then present.N.decoder.key, present.N.decoder.value (no encoder KV)
        var newEntries: [KVCacheEntry] = []
        for i in 0..<numLayers {
            let baseIdx = 1 + i * 2
            let dk = try extractTensorData(api: api, tensor: outputs[baseIdx]!)
            let dv = try extractTensorData(api: api, tensor: outputs[baseIdx + 1]!)
            // Reuse encoder KV from previous cache (unchanged)
            let ek = previousKVCache.entries[i].encoderKey
            let ev = previousKVCache.entries[i].encoderValue
            newEntries.append(KVCacheEntry(decoderKey: dk, decoderValue: dv, encoderKey: ek, encoderValue: ev))
        }

        for o in outputs { api.pointee.ReleaseValue(o) }

        return (nextToken, KVCache(entries: newEntries))
    }

    // MARK: - Tensor Data Extraction

    nonisolated private static func extractTensorData(
        api: UnsafePointer<OrtApi>,
        tensor: OpaquePointer
    ) throws -> TensorData {
        var data: UnsafeMutableRawPointer?
        try ortCheck(api, api.pointee.GetTensorMutableData(tensor, &data))

        var shapeInfo: OpaquePointer?
        try ortCheck(api, api.pointee.GetTensorTypeAndShape(tensor, &shapeInfo))
        defer { api.pointee.ReleaseTensorTypeAndShapeInfo(shapeInfo) }

        var dimCount: Int = 0
        try ortCheck(api, api.pointee.GetDimensionsCount(shapeInfo, &dimCount))
        var dims = [Int64](repeating: 0, count: dimCount)
        try ortCheck(api, api.pointee.GetDimensions(shapeInfo, &dims, dimCount))

        let totalCount = Int(dims.reduce(1, *))
        let values: [Float]
        if totalCount == 0 || data == nil {
            values = []
        } else {
            let floatPtr = data!.assumingMemoryBound(to: Float.self)
            values = Array(UnsafeBufferPointer(start: floatPtr, count: totalCount))
        }

        return TensorData(data: values, shape: dims.map { Int($0) })
    }
}

// MARK: - Data Types

// S3: Mark data types as private (file-private)
private struct EncoderOutput: Sendable {
    let data: [Float]
    let shape: [Int]  // [batch, seqLen, hiddenSize]
}

private struct TensorData: Sendable {
    let data: [Float]
    let shape: [Int]
}

private struct KVCacheEntry: Sendable {
    let decoderKey: TensorData
    let decoderValue: TensorData
    let encoderKey: TensorData
    let encoderValue: TensorData
}

private struct KVCache: Sendable {
    let entries: [KVCacheEntry]
}

// MARK: - C String Array Helper

private extension Array where Element == String {
    func withCArrayOfCStrings<R>(_ body: (UnsafeMutablePointer<UnsafePointer<CChar>?>) throws -> R) rethrows -> R {
        let cStrings = self.map { strdup($0) }
        defer { cStrings.forEach { free($0) } }
        var ptrs = cStrings.map { UnsafePointer<CChar>($0) as UnsafePointer<CChar>? }
        return try body(&ptrs)
    }
}
