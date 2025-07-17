package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LogSoftmaxTest {
    private fun getTargetPath(dirName: String) = "log_softmax/v1/$dirName/"

    @Test
    fun test_log_softmax_axis_0() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_0"))
    }

    @Test
    fun test_log_softmax_axis_1() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_1"))
    }

    @Test
    fun test_log_softmax_axis_2() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_2"))
    }

    @Test
    fun test_log_softmax_default_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_default_axis"))
    }

    @Test
    fun test_log_softmax_example_1() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_example_1"))
    }

    @Test
    fun test_log_softmax_large_number() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_large_number"))
    }

    @Test
    fun test_log_softmax_negative_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_negative_axis"))
    }
}
