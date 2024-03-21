package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ExpTest {
    private fun getTargetPath(dirName: String) = "exp/$dirName/"

    @Test
    fun test_exp() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_exp"))
    }

    @Test
    fun test_exp_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_exp_example"))
    }
}
