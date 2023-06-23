package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CoshTest {
    private fun getTargetPath(dirName: String) = "cosh/$dirName/"

    @Test
    fun test_cosh() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cosh"))
    }

    @Test
    fun test_cosh_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cosh_example"))
    }
}
