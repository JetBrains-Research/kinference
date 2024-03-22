package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AsinTest {
    private fun getTargetPath(dirName: String) = "asin/$dirName/"

    @Test
    fun test_asin() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_asin"))
    }

    @Test
    fun test_asin_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_asin_example"))
    }
}
