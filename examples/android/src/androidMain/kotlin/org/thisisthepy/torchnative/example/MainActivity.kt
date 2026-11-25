package org.thisisthepy.torchnative.example

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import python.multiplatform.env.PythonBootstrap
import python.multiplatform.ffi.Python3

/**
 * Runs the common task of torchnative's examples (examples/README.md) on this phone and shows it.
 *
 * `adb shell am start -n org.thisisthepy.torchnative.example/.MainActivity --es device cpu`
 * (or `npu`) picks the device; the default is `cpu`. Every streamed piece of text and the final
 * JSON line (device, text, tokens per second, whether the ids equal upstream's) go to logcat
 * under [TAG], so a run can be read without the screen:
 * `adb logcat -s TorchnativeExample`. The activity finishes itself after a headless run
 * (`--ez headless true`).
 */
class MainActivity : Activity() {
    private lateinit var output: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        output = TextView(this).apply { setPadding(32, 32, 32, 32); textSize = 15f }
        setContentView(ScrollView(this).apply { addView(output) })

        val device = intent?.getStringExtra(DEVICE_EXTRA) ?: "cpu"
        val headless = intent?.getBooleanExtra(HEADLESS_EXTRA, false) == true
        show("torchnative example: SmolLM2-135M, greedy, 32 tokens, device=$device\n\n")
        Thread {
            val ok = runCommonTask(device)
            if (headless) runOnUiThread { finish() }
            Log.i(TAG, if (ok) "COMMON_TASK_DONE" else "COMMON_TASK_FAILED")
        }.start()
    }

    private fun show(text: String) = runOnUiThread { output.append(text) }

    private fun runCommonTask(device: String): Boolean =
        try {
            PythonBootstrap.initialize(this)
            // The model goes to the app's own files directory, not a shared location.
            val hfHome = filesDir.resolve("hf-home").absolutePath
            Python3.exec(
                """
                |import os
                |os.environ["HF_HOME"] = ${pyString(hfHome)}
                |from tn_example import common_task, android_log
                |_tn_result = common_task.run_json(${pyString(device)}, emit=android_log.log)
                |android_log.log("COMMON_TASK " + _tn_result)
                """.trimMargin(),
            )
            val result = Python3.eval("_tn_result").toString()
            show("\n\n$result\n")
            true
        } catch (t: Throwable) {
            Log.e(TAG, "COMMON_TASK_ERROR: $t", t)
            show("\n\nFAILED: $t\n")
            false
        }

    private fun pyString(value: String): String =
        "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    companion object {
        const val TAG = "TorchnativeExample"
        const val DEVICE_EXTRA = "device"
        const val HEADLESS_EXTRA = "headless"
    }
}
