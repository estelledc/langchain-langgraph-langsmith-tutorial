import CoreGraphics

enum PanelHeightSource: Equatable {
    case systemKeyboard(notificationHeight: CGFloat, guideOverlap: CGFloat)
    case businessPanel(CGFloat)
    case hidden
}

struct KeyboardHeightResolver {
    func resolve(_ source: PanelHeightSource) -> CGFloat {
        switch source {
        case let .systemKeyboard(notificationHeight, _):
            // XCODEFIX_BUG: a notification is an event snapshot, not the final geometry source.
            return max(0, notificationHeight)
        case let .businessPanel(height):
            return max(0, height)
        case .hidden:
            return 0
        }
    }
}
