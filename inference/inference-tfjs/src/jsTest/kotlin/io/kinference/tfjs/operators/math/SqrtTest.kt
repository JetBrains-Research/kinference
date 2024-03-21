package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SqrtTest {
    private fun getTargetPath(dirName: String) = "sqrt/$dirName/"

    @Test
    fun test_sqrt() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_sqrt"))
    }

    @Test
    fun test_sqrt_example() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_sqrt_example"))
    }
}
