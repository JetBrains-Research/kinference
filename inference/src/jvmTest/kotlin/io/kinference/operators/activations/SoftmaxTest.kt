package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class SoftmaxTest {
    private fun getTargetPath(dirName: String) = "/softmax/$dirName/"

    @Test
    fun `test softmax (axis=0)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_0"))
    }

    @Test
    fun `test softmax (axis=1)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_1"))
    }

    @Test
    fun `test softmax (axis=2)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_2"))
    }

    @Test
    fun `test softmax with default axis`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_softmax_default_axis"))
    }

    @Test
    fun `test softmax with large number`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_softmax_large_number"))
    }

    @Test
    fun `test softmax with negative axis`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_softmax_negative_axis"))
    }
}
