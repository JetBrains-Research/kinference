package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SoftmaxTest {
    private fun getTargetPath(dirName: String) = "softmax/$dirName/"

    @Test
    fun test_softmax_axis_0() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_0"))
    }

    @Test
    fun test_softmax_axis_1() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_1"))
    }

    @Test
    fun test_softmax_axis_2() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_axis_2"))
    }

    @Test
    fun test_softmax_with_default_axis() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_default_axis"))
    }

    @Test
    fun test_softmax_with_large_number() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_large_number"))
    }

    @Test
    fun test_softmax_with_negative_axis() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_softmax_negative_axis"))
    }
}
