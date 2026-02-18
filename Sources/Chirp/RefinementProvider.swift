import Foundation

public enum RefinementProvider: String, CaseIterable, Sendable {
    case none
    case t5Local = "t5-local"
    case claude = "claude"

    public var displayName: String {
        switch self {
        case .none: return "None"
        case .t5Local: return "Flan-T5 (local)"
        case .claude: return "Claude (cloud)"
        }
    }

    // MARK: - Persistence
    static let defaultsKey = "refinementProvider"

    public static var saved: RefinementProvider {
        get {
            guard let raw = UserDefaults.standard.string(forKey: defaultsKey),
                  let provider = RefinementProvider(rawValue: raw) else {
                return .none
            }
            return provider
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: defaultsKey)
        }
    }
}
