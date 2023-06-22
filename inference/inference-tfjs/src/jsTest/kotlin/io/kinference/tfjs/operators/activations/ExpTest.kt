package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ExpTest {
    private fun getTargetPath(dirName: String) = "exp/$dirName/"

    @Test
    fun test_exp() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_exp"))
    }

    @Test
    fun test_exp_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_exp_example"))
    }
}
