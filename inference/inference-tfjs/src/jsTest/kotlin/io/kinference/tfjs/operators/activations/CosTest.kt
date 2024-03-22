package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CosTest {
    private fun getTargetPath(dirName: String) = "cos/$dirName/"

    @Test
    fun test_cos() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cos"))
    }

    @Test
    fun test_cos_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cos_example"))
    }
}
