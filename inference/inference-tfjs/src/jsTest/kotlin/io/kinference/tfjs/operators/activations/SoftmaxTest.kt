package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SoftmaxTest {
    private fun getTargetPath(dirName: String) = "softmax/$dirName/"

    @Test
    fun test_softmax_axis_2() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_2"))
    }

    @Test
    fun test_softmax_with_large_number() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_softmax_large_number"))
    }

    @Test
    fun test_softmax_with_negative_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_softmax_negative_axis"))
    }
}
