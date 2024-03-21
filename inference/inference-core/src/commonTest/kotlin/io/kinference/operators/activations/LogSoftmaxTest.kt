package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LogSoftmaxTest {
    private fun getTargetPath(dirName: String) = "log_softmax/$dirName/"

    @Test
    fun test_log_softmax_axis_0() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_0"))
    }

    @Test
    fun test_log_softmax_axis_1() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_1"))
    }

    @Test
    fun test_log_softmax_axis_2() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_2"))
    }

    @Test
    fun test_log_softmax_default_axis() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_default_axis"))
    }

    @Test
    fun test_log_softmax_example_1() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_example_1"))
    }

    @Test
    fun test_log_softmax_large_number() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_large_number"))
    }

    @Test
    fun test_log_softmax_negative_axis() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_negative_axis"))
    }
}
