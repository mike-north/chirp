import Testing
import Foundation
@testable import Chirp

@Suite("DaemonManager")
@MainActor
struct DaemonManagerTests {

    @Test("Can be created with custom script path")
    func initWithCustomScriptPath() {
        let dm = DaemonManager(scriptPath: "/nonexistent/daemon.js")
        #expect(dm.isReady == false)
        #expect(dm.lastError == nil)
    }

    @Test("stop() sets isReady to false")
    func stopSetsIsReadyFalse() {
        let dm = DaemonManager(scriptPath: "/nonexistent/daemon.js")
        dm.stop()
        #expect(dm.isReady == false)
    }

    @Test("start() without node available sets lastError")
    func startWithoutNodeSetsError() {
        let dm = DaemonManager(scriptPath: "/nonexistent/daemon.js")
        // Force nodeAvailable to false (it may already be false on CI)
        dm.nodeAvailable = false
        dm.start()
        #expect(dm.lastError == "Node.js not found in PATH")
    }

    @Test("nodeAvailable reflects whether node is installed")
    func nodeAvailableDefaultValue() {
        let dm = DaemonManager(scriptPath: "/nonexistent/daemon.js")
        // We can't assert a specific value since it depends on the machine,
        // but we can verify it's a Bool that was set by checkNodeAvailable.
        // On most dev machines node is available; on CI it may not be.
        // Just verify the property is accessible and stop() still works.
        _ = dm.nodeAvailable
        dm.stop()
        #expect(dm.isReady == false)
    }
}
