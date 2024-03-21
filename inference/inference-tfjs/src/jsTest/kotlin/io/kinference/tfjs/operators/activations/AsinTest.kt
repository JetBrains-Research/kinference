package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AsinTest {
    private fun getTargetPath(dirName: String) = "asin/$dirName/"

    @Test
    fun test_asin() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_asin"))
    }

    @Test
    fun test_asin_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_asin_example"))
    }
}
