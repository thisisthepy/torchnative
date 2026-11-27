package org.thisisthepy.torchnative.example

import kotlinx.cinterop.BetaInteropApi
import kotlinx.cinterop.ExperimentalForeignApi
import platform.Foundation.NSCachesDirectory
import platform.Foundation.NSDocumentDirectory
import platform.Foundation.NSProcessInfo
import platform.Foundation.NSSearchPathForDirectoriesInDomains
import platform.Foundation.NSString
import platform.Foundation.NSUTF8StringEncoding
import platform.Foundation.NSUserDomainMask
import platform.Foundation.create
import platform.Foundation.writeToFile
import platform.posix.exit
import python.multiplatform.ffi.Python3

/**
 * Runs the common task of torchnative's examples (examples/README.md) in the iOS app and exits.
 *
 * `xcrun simctl launch <device> org.thisisthepy.torchnative.example --common-task cpu` (or `npu`)
 * picks the device. The result, the same JSON line the other examples print, goes to stdout and to
 * [COMMON_TASK_FILE] in the app's Documents directory, which CI reads with
 * `simctl get_app_container <device> <bundle> data`: the console is not reliable once the app has
 * exited (toolchain#92, which this follows). Without the argument this does nothing.
 */
const val COMMON_TASK_ARGUMENT = "--common-task"

/** The report's file name in the app's Documents directory. */
const val COMMON_TASK_FILE = "common-task.txt"

private const val PY_EVAL_INPUT = 258

fun runCommonTaskIfRequested() {
    val arguments = NSProcessInfo.processInfo.arguments.map { it.toString() }
    val at = arguments.indexOf(COMMON_TASK_ARGUMENT)
    if (at < 0) return
    val device = arguments.getOrNull(at + 1) ?: "cpu"
    val (code, report) =
        try {
            Python3.initialize()
            // The model goes to the app's own Caches directory: HF's default under HOME is not
            // writable in an iOS app.
            val hfHome = "${directory(NSCachesDirectory)}/hf-home"
            Python3.exec(
                """
                |import os
                |os.environ["HF_HOME"] = ${pyString(hfHome)}
                |from tn_example import common_task
                |_tn_result = common_task.run_json(${pyString(device)}, emit=lambda text: None)
                """.trimMargin(),
            )
            val globals = Python3.import("__main__").dict
            0 to "COMMON_TASK " + Python3.eval("_tn_result", PY_EVAL_INPUT, globals, globals).toString()
        } catch (t: Throwable) {
            1 to "COMMON_TASK_FAILED: $t"
        }
    println(report)
    writeReport("$report\n")
    exit(code)
}

/** A Python string literal for [value]; the paths and device names here hold no quote or backslash. */
private fun pyString(value: String): String {
    require('"' !in value && '\\' !in value) { "cannot quote $value for Python" }
    return "\"$value\""
}

private fun directory(kind: ULong): String =
    NSSearchPathForDirectoriesInDomains(kind, NSUserDomainMask, true).firstOrNull() as? String
        ?: error("no directory of kind $kind for this app")

@OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
private fun writeReport(report: String) {
    NSString.create(string = report).writeToFile(
        "${directory(NSDocumentDirectory)}/$COMMON_TASK_FILE",
        atomically = true,
        encoding = NSUTF8StringEncoding,
        error = null,
    )
}
