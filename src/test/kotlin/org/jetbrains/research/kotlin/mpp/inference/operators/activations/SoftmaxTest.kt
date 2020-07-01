package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class SoftmaxTest {
    private fun getTargetPath(dirName: String) = "/softmax/$dirName/"

    @Test
    fun test_softmax_axis_0() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_axis_0"))
    }

    @Test
    fun test_softmax_axis_1() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_axis_1"))
    }

    @Test
    fun test_softmax_axis_2() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_axis_2"))
    }

    @Test
    fun test_softmax_default_axis() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_default_axis"))
    }

    @Test
    fun test_softmax_large_number() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_large_number"))
    }

    @Test
    fun test_softmax_negative_axis() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_negative_axis"))
    }
}
