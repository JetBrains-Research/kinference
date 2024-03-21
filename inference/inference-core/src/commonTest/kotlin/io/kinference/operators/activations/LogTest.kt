package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LogTest {
    private fun getTargetPath(dirName: String) = "log/$dirName/"

    @Test
    fun test_log() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_log"))
    }

    @Test
    fun test_log_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_log_example"))
    }
}
