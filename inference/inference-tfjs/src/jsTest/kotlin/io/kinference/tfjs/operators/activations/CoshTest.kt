package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CoshTest {
    private fun getTargetPath(dirName: String) = "cosh/$dirName/"

    @Test
    fun test_cosh() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cosh"))
    }

    @Test
    fun test_cosh_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cosh_example"))
    }
}
