package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LogSoftmaxVer13Test {
    private fun getTargetPath(dirName: String) = "log_softmax/v13/$dirName/"

    @Test
    fun test_log_softmax_axis_0() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_0"))
    }

    @Test
    fun test_log_softmax_axis_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_1"))
    }

    @Test
    fun test_log_softmax_axis_2() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_2"))
    }

    @Test
    fun test_log_softmax_default_axis() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_default_axis"))
    }

    @Test
    fun test_log_softmax_example_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_example_1"))
    }

    @Test
    fun test_log_softmax_large_number() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_large_number"))
    }

    @Test
    fun test_log_softmax_negative_axis() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_negative_axis"))
    }
}

class LogSoftmaxVer1Test {
    private fun getTargetPath(dirName: String) = "log_softmax/v1/$dirName/"

    @Test
    fun test_log_softmax_axis_0() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_0"))
    }

    @Test
    fun test_log_softmax_axis_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_1"))
    }

    @Test
    fun test_log_softmax_axis_2() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_axis_2"))
    }

    @Test
    fun test_log_softmax_default_axis() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_default_axis"))
    }

    @Test
    fun test_log_softmax_example_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_example_1"))
    }

    @Test
    fun test_log_softmax_large_number() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_large_number"))
    }

    @Test
    fun test_log_softmax_negative_axis() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_logsoftmax_negative_axis"))
    }
}
