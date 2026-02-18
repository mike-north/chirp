// DaemonManager.swift — Manages the chirp-daemon Node.js process lifecycle.
// Spawns the daemon, monitors socket creation, and auto-restarts on failure
// with exponential backoff.

import AppKit

@Observable
@MainActor
public final class DaemonManager {
    public var isReady = false
    public var lastError: String?
    public var nodeAvailable = false

    private var process: Process?
    private var monitorTask: Task<Void, Never>?
    private var stopped = false
    private var restartCount = 0
    private let maxBackoff: TimeInterval = 30
    private let socketPath = NSTemporaryDirectory() + "chirp-claude.sock"
    private var nodePath: String?

    private static let commonNodePaths = [
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",
    ]

    /// Path to the daemon script (index.js).
    private let scriptPath: String

    init(scriptPath: String? = nil) {
        if let path = scriptPath {
            self.scriptPath = path
        } else {
            // Try bundled path first, then fall back to dev path relative to the executable.
            let candidates: [String] = {
                var paths = [String]()
                if let res = Bundle.main.resourcePath {
                    paths.append(res + "/chirp-daemon/dist/index.js")
                }
                // Dev: project root relative to executable (bazel run puts binary in sandbox,
                // but we can locate the workspace via BUILD_WORKSPACE_DIRECTORY or cwd).
                if let ws = ProcessInfo.processInfo.environment["BUILD_WORKSPACE_DIRECTORY"] {
                    paths.append(ws + "/chirp-daemon/dist/index.js")
                }
                // Fallback: current working directory
                paths.append(FileManager.default.currentDirectoryPath + "/chirp-daemon/dist/index.js")
                return paths
            }()
            self.scriptPath = candidates.first { FileManager.default.fileExists(atPath: $0) }
                ?? candidates.last!
        }
        checkNodeAvailable()

        // Ensure daemon is terminated when the app quits.
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.process?.terminate()
            }
        }
    }

    public func start() {
        guard nodeAvailable else {
            lastError = "Node.js not found in PATH"
            return
        }
        stopped = false
        spawnDaemon()
    }

    public func stop() {
        stopped = true
        monitorTask?.cancel()
        monitorTask = nil
        if let proc = process, proc.isRunning {
            proc.terminate()
        }
        process = nil
        isReady = false
    }

    // MARK: - Private

    private func checkNodeAvailable() {
        // First try `which node` (works when PATH is set, e.g. terminal launches).
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        proc.arguments = ["node"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do {
            try proc.run()
            proc.waitUntilExit()
            if proc.terminationStatus == 0,
               let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                   .trimmingCharacters(in: .whitespacesAndNewlines),
               !output.isEmpty {
                nodePath = output
                nodeAvailable = true
                return
            }
        } catch {
            // Fall through to manual search.
        }

        // GUI apps have a minimal PATH. Check common install locations.
        var candidates = Self.commonNodePaths

        // Check nvm installations.
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let nvmVersions = home + "/.nvm/versions/node"
        if let versions = try? FileManager.default.contentsOfDirectory(atPath: nvmVersions) {
            // Sort descending so we pick the latest version.
            let sorted = versions.sorted { $0.compare($1, options: .numeric) == .orderedDescending }
            for version in sorted {
                candidates.append(nvmVersions + "/" + version + "/bin/node")
            }
        }

        for path in candidates {
            if FileManager.default.isExecutableFile(atPath: path) {
                nodePath = path
                nodeAvailable = true
                return
            }
        }

        nodeAvailable = false
    }

    private func spawnDaemon() {
        try? FileManager.default.removeItem(atPath: socketPath)

        NSLog("[DaemonManager] Spawning daemon at: %@", scriptPath)

        let proc = Process()
        if let nodePath {
            proc.executableURL = URL(fileURLWithPath: nodePath)
            proc.arguments = [scriptPath]
        } else {
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            proc.arguments = ["node", scriptPath]
        }
        proc.environment = ProcessInfo.processInfo.environment
        let errPipe = Pipe()
        proc.standardError = errPipe
        proc.standardOutput = Pipe()

        // Read stderr asynchronously to capture daemon errors
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                NSLog("[chirp-daemon] %@", str.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }

        proc.terminationHandler = { @Sendable [weak self] p in
            let status = p.terminationStatus
            errPipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor [weak self] in
                self?.handleTermination(status: status)
            }
        }

        do {
            try proc.run()
            self.process = proc
            lastError = nil
            startMonitoring()
        } catch {
            lastError = "Failed to start daemon: \(error.localizedDescription)"
        }
    }

    private func startMonitoring() {
        monitorTask?.cancel()
        monitorTask = Task {
            // Poll for socket creation with 100ms intervals for up to 10 seconds (100 × 100ms).
            for _ in 0..<100 {
                if Task.isCancelled { return }
                if FileManager.default.fileExists(atPath: socketPath) {
                    isReady = true
                    restartCount = 0
                    return
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
            if !Task.isCancelled {
                lastError = "Daemon socket not created within timeout"
            }
        }
    }

    private func handleTermination(status: Int32) {
        isReady = false
        process = nil

        guard !stopped else { return }

        restartCount += 1
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s, then capped at 30s.
        let delay = min(pow(2.0, Double(restartCount - 1)), maxBackoff)
        lastError = "Daemon exited (status \(status)), restarting in \(Int(delay))s..."

        monitorTask = Task {
            try? await Task.sleep(for: .seconds(delay))
            if !Task.isCancelled {
                spawnDaemon()
            }
        }
    }
}
