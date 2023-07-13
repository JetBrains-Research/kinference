package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TanTest {
    private fun getTargetPath(dirName: String) = "tan/$dirName/"

    @Test
    fun test_tanh_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tan_example"))
    }

    @Test
    fun test_tanh() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tan"))
    }
}
