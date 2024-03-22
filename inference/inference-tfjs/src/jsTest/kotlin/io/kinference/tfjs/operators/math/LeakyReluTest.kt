package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LeakyReluTest {
    private fun getTargetPath(dirName: String) = "leakyrelu/$dirName/"

    @Test
    fun test_leaky_relu() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu"))
    }

    @Test
    fun test_leaky_relu_default() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu_default"))
    }

    @Test
    fun test_leaky_relu_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu_example"))
    }
}
