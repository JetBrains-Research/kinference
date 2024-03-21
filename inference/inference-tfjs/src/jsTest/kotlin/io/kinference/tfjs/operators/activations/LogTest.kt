package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LogTest {
    private fun getTargetPath(dirName: String) = "log/$dirName/"

    @Test
    fun test_log() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_log"))
    }

    @Test
    fun test_log_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_log_example"))
    }
}
