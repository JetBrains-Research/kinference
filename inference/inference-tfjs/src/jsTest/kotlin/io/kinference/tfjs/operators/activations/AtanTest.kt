package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AtanTest {
    private fun getTargetPath(dirName: String) = "atan/$dirName/"

    @Test
    fun test_atan() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_atan"))
    }

    @Test
    fun test_atan_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_atan_example"))
    }
}
