package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SqrtTest {
    private fun getTargetPath(dirName: String) = "sqrt/$dirName/"

    @Test
    fun test_sqrt() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_sqrt"))
    }

    @Test
    fun test_sqrt_example() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_sqrt_example"))
    }
}
