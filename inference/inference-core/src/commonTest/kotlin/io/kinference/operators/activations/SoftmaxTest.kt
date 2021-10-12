package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SoftmaxTest {
    private fun getTargetPath(dirName: String) = "/softmax/$dirName/"

    @Test
    fun test_softmax_axis_0() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_0"))
    }

    @Test
    fun test_softmax_axis_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_1"))
    }

    @Test
    fun test_softmax_axis_2() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_2"))
    }

    @Test
    fun test_softmax_with_default_axis() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_default_axis"))
    }

    @Test
    fun test_softmax_with_large_number() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_large_number"))
    }

    @Test
    fun test_softmax_with_negative_axis() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_negative_axis"))
    }
}
