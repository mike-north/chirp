// T5ModelManager.swift — Downloads T5-small ONNX model files from HuggingFace.
// Downloads encoder, decoder, and tokenizer files sequentially, tracking
// aggregate progress across all files. Stores files in Application Support.

import Foundation

final class T5ModelManager: NSObject, @unchecked Sendable, URLSessionDownloadDelegate {
    private let onProgress: @Sendable (Double) -> Void
    private let onComplete: @Sendable (Result<T5ModelPaths, Error>) -> Void
    private var session: URLSession?

    /// Files to download, ordered by expected size (largest first for better progress UX).
    /// Uses split decoder models (no-past + with-past) instead of merged to avoid
    /// ORT If-node shape validation issues with the merged model.
    private static let files: [(name: String, url: URL, weight: Double)] = [
        (
            "decoder_model_quantized.onnx",
            URL(string: "https://huggingface.co/Xenova/flan-t5-small/resolve/main/onnx/decoder_model_quantized.onnx")!,
            0.35
        ),
        (
            "decoder_with_past_model_quantized.onnx",
            URL(string: "https://huggingface.co/Xenova/flan-t5-small/resolve/main/onnx/decoder_with_past_model_quantized.onnx")!,
            0.30
        ),
        (
            "encoder_model_quantized.onnx",
            URL(string: "https://huggingface.co/Xenova/flan-t5-small/resolve/main/onnx/encoder_model_quantized.onnx")!,
            0.25
        ),
        (
            "tokenizer.json",
            URL(string: "https://huggingface.co/Xenova/flan-t5-small/resolve/main/tokenizer.json")!,
            0.05
        ),
        (
            "tokenizer_config.json",
            URL(string: "https://huggingface.co/Xenova/flan-t5-small/resolve/main/tokenizer_config.json")!,
            0.03
        ),
        (
            "config.json",
            URL(string: "https://huggingface.co/Xenova/flan-t5-small/resolve/main/config.json")!,
            0.02
        ),
    ]

    /// Check file — if this exists, the download is considered complete.
    private static let checkFile = "encoder_model_quantized.onnx"

    private var currentFileIndex = 0
    private var completedWeight: Double = 0

    init(
        onProgress: @escaping @Sendable (Double) -> Void,
        onComplete: @escaping @Sendable (Result<T5ModelPaths, Error>) -> Void
    ) {
        self.onProgress = onProgress
        self.onComplete = onComplete
    }

    // MARK: - Model discovery

    static func findExisting() -> T5ModelPaths? {
        let modelDir = installDir
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(modelDir)/\(checkFile)") else { return nil }
        return T5ModelPaths(
            encoderPath: "\(modelDir)/encoder_model_quantized.onnx",
            decoderPath: "\(modelDir)/decoder_model_quantized.onnx",
            decoderWithPastPath: "\(modelDir)/decoder_with_past_model_quantized.onnx",
            tokenizerDir: modelDir
        )
    }

    static var installDir: String {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!.appendingPathComponent("Chirp").path
        return "\(appSupport)/models/flan-t5-small"
    }

    // MARK: - Download

    func download() {
        currentFileIndex = 0
        completedWeight = 0
        downloadNextFile()
    }

    func cancel() {
        session?.invalidateAndCancel()
        session = nil
    }

    private func downloadNextFile() {
        guard currentFileIndex < Self.files.count else {
            // All files downloaded — verify and complete
            guard let paths = Self.findExisting() else {
                onComplete(.failure(T5DownloadError.modelFilesNotFound))
                return
            }
            onProgress(1.0)
            onComplete(.success(paths))
            return
        }

        let config = URLSessionConfiguration.default
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        let file = Self.files[currentFileIndex]
        let task = session!.downloadTask(with: file.url)
        task.resume()
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let file = Self.files[currentFileIndex]
        let fileProgress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        let overall = completedWeight + fileProgress * file.weight
        onProgress(overall)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let installDir = Self.installDir
        let fm = FileManager.default
        do {
            try fm.createDirectory(atPath: installDir, withIntermediateDirectories: true)
        } catch {
            onComplete(.failure(error))
            return
        }

        let file = Self.files[currentFileIndex]
        let destPath = "\(installDir)/\(file.name)"

        // Remove existing file if present (re-download case)
        if fm.fileExists(atPath: destPath) {
            try? fm.removeItem(atPath: destPath)
        }

        do {
            try fm.moveItem(atPath: location.path, toPath: destPath)
        } catch {
            onComplete(.failure(error))
            return
        }

        completedWeight += file.weight
        currentFileIndex += 1
        downloadNextFile()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            if (error as NSError).domain == NSURLErrorDomain,
               (error as NSError).code == NSURLErrorCancelled {
                return
            }
            onComplete(.failure(error))
        }
    }

    enum T5DownloadError: LocalizedError {
        case modelFilesNotFound

        var errorDescription: String? {
            switch self {
            case .modelFilesNotFound: return "T5 model files not found after download"
            }
        }
    }
}
