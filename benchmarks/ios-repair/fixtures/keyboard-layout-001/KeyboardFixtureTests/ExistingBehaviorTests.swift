import XCTest
@testable import KeyboardFixture

final class ExistingBehaviorTests: XCTestCase {
    private let resolver = KeyboardHeightResolver()

    func testInitialKeyboardSnapshotMatchesInitialGuide() {
        let height = resolver.resolve(
            .systemKeyboard(notificationHeight: 325, guideOverlap: 325)
        )
        XCTAssertEqual(height, 325)
    }

    func testBusinessPanelKeepsItsExplicitHeight() {
        XCTAssertEqual(resolver.resolve(.businessPanel(384)), 384)
    }

    func testHiddenPanelCollapsesToZero() {
        XCTAssertEqual(resolver.resolve(.hidden), 0)
    }
}
