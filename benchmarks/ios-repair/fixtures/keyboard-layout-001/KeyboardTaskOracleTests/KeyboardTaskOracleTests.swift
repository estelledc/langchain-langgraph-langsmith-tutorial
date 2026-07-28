import XCTest
@testable import KeyboardFixture

final class KeyboardTaskOracleTests: XCTestCase {
    private let resolver = KeyboardHeightResolver()

    func testPanelFollowsDelayedGuideWithoutSecondNotification() {
        let height = resolver.resolve(
            .systemKeyboard(notificationHeight: 325, guideOverlap: 521)
        )
        XCTAssertEqual(height, 521)
    }

    func testFixDoesNotHardCodeTheKnownFinalHeight() {
        let height = resolver.resolve(
            .systemKeyboard(notificationHeight: 325, guideOverlap: 384)
        )
        XCTAssertEqual(height, 384)
    }
}
