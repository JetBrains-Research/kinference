package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AtanTest {
    private fun getTargetPath(dirName: String) = "atan/$dirName/"

    @Test
    fun test_atan() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_atan"))
    }

    @Test
    fun test_atan_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_atan_example"))
    }
}
