import SwiftUI
import Project

@main
struct iOSApp: App {
    init() {
        // `xcrun simctl launch <device> <bundle-id> --common-task cpu` runs the common task and exits
        // before any UI (CommonTask.ios.kt); without the argument the app only shows ContentView.
        CommonTask_iosKt.runCommonTaskIfRequested()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
