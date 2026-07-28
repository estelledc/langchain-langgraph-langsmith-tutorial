import UIKit

final class KeyboardFixtureViewController: UIViewController {
    private enum Constants {
        static let notificationHeight: CGFloat = 325
        static let finalGuideOverlap: CGFloat = 521
        static let tolerance: CGFloat = 0.5
        static let resultFile = "xcodefix-result.json"
    }

    private struct RuntimeRecord: Codable {
        let phase: String
        let outcome: String
        let notificationCount: Int
        let notificationHeight: Double
        let guideMinY: Double
        let guideOverlap: Double
        let panelHeight: Double
        let mismatch: Double
    }

    private let resolver = KeyboardHeightResolver()
    private let simulatedKeyboardGuide = UILayoutGuide()
    private let panel = UIView()
    private let guideLine = UIView()
    private let mismatchView = UIView()
    private let titleLabel = UILabel()
    private let metricsLabel = UILabel()
    private let statusLabel = UILabel()
    private var guideHeightConstraint: NSLayoutConstraint!
    private var panelHeightConstraint: NSLayoutConstraint!
    private var didAutorun = false
    private var notificationCount = 0
    private var notificationHeight: CGFloat = 0

    override func viewDidLoad() {
        super.viewDidLoad()
        configureView()
        configureLayout()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        guard !didAutorun, ProcessInfo.processInfo.arguments.contains("--autorun") else { return }
        didAutorun = true
        removePreviousResult()
        DispatchQueue.main.async { [weak self] in
            self?.runDeterministicScenario()
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        updateOverlayFrames()
    }

    private func configureView() {
        view.backgroundColor = .systemBackground

        titleLabel.text = "XcodeFixBench · Keyboard Layout 001"
        titleLabel.font = .boldSystemFont(ofSize: 22)
        titleLabel.numberOfLines = 0

        metricsLabel.font = .monospacedSystemFont(ofSize: 16, weight: .regular)
        metricsLabel.numberOfLines = 0
        metricsLabel.accessibilityIdentifier = "runtimeMetrics"

        statusLabel.font = .boldSystemFont(ofSize: 20)
        statusLabel.numberOfLines = 0
        statusLabel.accessibilityIdentifier = "runtimeOutcome"
        statusLabel.text = "Preparing deterministic replay…"

        panel.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.24)
        guideLine.backgroundColor = .systemGreen
        mismatchView.backgroundColor = UIColor.systemRed.withAlphaComponent(0.28)
        mismatchView.isUserInteractionEnabled = false
    }

    private func configureLayout() {
        view.addLayoutGuide(simulatedKeyboardGuide)
        guideHeightConstraint = simulatedKeyboardGuide.heightAnchor.constraint(equalToConstant: 0)
        panelHeightConstraint = panel.heightAnchor.constraint(equalToConstant: 0)

        [panel, titleLabel, metricsLabel, statusLabel].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview($0)
        }
        view.addSubview(mismatchView)
        view.addSubview(guideLine)

        NSLayoutConstraint.activate([
            simulatedKeyboardGuide.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            simulatedKeyboardGuide.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            simulatedKeyboardGuide.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            guideHeightConstraint,

            panel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            panel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            panel.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            panelHeightConstraint,

            titleLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 32),
            titleLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            titleLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),

            metricsLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 28),
            metricsLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            metricsLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),

            statusLabel.topAnchor.constraint(equalTo: metricsLabel.bottomAnchor, constant: 28),
            statusLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),
        ])
    }

    private func runDeterministicScenario() {
        notificationCount = 1
        notificationHeight = Constants.notificationHeight
        guideHeightConstraint.constant = Constants.notificationHeight
        updatePanelHeight()
        view.layoutIfNeeded()
        updateMetrics()

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self] in
            guard let self else { return }
            self.guideHeightConstraint.constant = Constants.finalGuideOverlap
            self.view.layoutIfNeeded()
            self.updatePanelHeight()
            self.view.layoutIfNeeded()
            self.finishReplay()
        }
    }

    private func updatePanelHeight() {
        let source = PanelHeightSource.systemKeyboard(
            notificationHeight: notificationHeight,
            guideOverlap: currentGuideOverlap()
        )
        panelHeightConstraint.constant = resolver.resolve(source)
    }

    private func finishReplay() {
        updateMetrics()
        let guideOverlap = currentGuideOverlap()
        let panelHeight = panel.bounds.height
        let mismatch = abs(guideOverlap - panelHeight)
        let outcome: String

        if mismatch < Constants.tolerance,
           notificationCount == 1,
           abs(guideOverlap - Constants.finalGuideOverlap) < Constants.tolerance {
            outcome = "fix_verified"
            statusLabel.text = "FIX VERIFIED · panel follows final guide geometry"
            statusLabel.textColor = .systemGreen
        } else if abs(panelHeight - Constants.notificationHeight) < Constants.tolerance,
                  notificationCount == 1,
                  abs(guideOverlap - Constants.finalGuideOverlap) < Constants.tolerance {
            outcome = "bug_reproduced"
            statusLabel.text = "BUG REPRODUCED · stale notification snapshot"
            statusLabel.textColor = .systemRed
        } else {
            outcome = "unexpected"
            statusLabel.text = "UNEXPECTED · oracle could not classify the result"
            statusLabel.textColor = .systemOrange
        }
        statusLabel.accessibilityValue = outcome

        let record = RuntimeRecord(
            phase: "complete",
            outcome: outcome,
            notificationCount: notificationCount,
            notificationHeight: Double(notificationHeight),
            guideMinY: Double(simulatedKeyboardGuide.layoutFrame.minY),
            guideOverlap: Double(guideOverlap),
            panelHeight: Double(panelHeight),
            mismatch: Double(mismatch)
        )
        write(record)
    }

    private func currentGuideOverlap() -> CGFloat {
        max(0, view.bounds.maxY - simulatedKeyboardGuide.layoutFrame.minY)
    }

    private func updateMetrics() {
        let guideOverlap = currentGuideOverlap()
        let panelHeight = panel.bounds.height
        metricsLabel.text = String(
            format: "notifications = %d\nnotification = %.0f\nguide overlap = %.0f\npanel = %.0f\nmismatch = %.0f",
            notificationCount,
            notificationHeight,
            guideOverlap,
            panelHeight,
            abs(guideOverlap - panelHeight)
        )
        updateOverlayFrames()
    }

    private func updateOverlayFrames() {
        let guideTop = simulatedKeyboardGuide.layoutFrame.minY
        let panelTop = panel.frame.minY
        let top = min(guideTop, panelTop)
        let bottom = max(guideTop, panelTop)
        mismatchView.frame = CGRect(x: 0, y: top, width: view.bounds.width, height: bottom - top)
        guideLine.frame = CGRect(x: 0, y: guideTop - 1, width: view.bounds.width, height: 2)
        view.bringSubviewToFront(mismatchView)
        view.bringSubviewToFront(guideLine)
        view.bringSubviewToFront(titleLabel)
        view.bringSubviewToFront(metricsLabel)
        view.bringSubviewToFront(statusLabel)
    }

    private func resultURL() -> URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent(Constants.resultFile)
    }

    private func removePreviousResult() {
        guard let url = resultURL() else { return }
        try? FileManager.default.removeItem(at: url)
    }

    private func write(_ record: RuntimeRecord) {
        guard let url = resultURL() else { return }
        do {
            let data = try JSONEncoder().encode(record)
            try data.write(to: url, options: .atomic)
            if let text = String(data: data, encoding: .utf8) {
                print("[XcodeFixBench] \(text)")
            }
        } catch {
            statusLabel.text = "INFRA ERROR · could not write runtime evidence"
            statusLabel.textColor = .systemOrange
            print("[XcodeFixBench] result write failed: \(error)")
        }
    }
}
