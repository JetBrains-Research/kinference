package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class SoftmaxTest {
    private fun getTargetPath(dirName: String) = "/softmax/$dirName/"

    @Test
    fun test_softmax_axis_0() {
        Utils.singleTestHelper(getTargetPath("test_softmax_axis_0"))
    }

    @Test
    fun test_softmax_axis_1() {
        Utils.singleTestHelper(getTargetPath("test_softmax_axis_1"))
    }

    @Test
    fun test_softmax_axis_2() {
        Utils.singleTestHelper(getTargetPath("test_softmax_axis_2"))
    }

    @Test
    fun test_softmax_default_axis() {
        Utils.singleTestHelper(getTargetPath("test_softmax_default_axis"))
    }

    @Test
    fun test_softmax_large_number() {
        Utils.singleTestHelper(getTargetPath("test_softmax_large_number"))
    }

    @Test
    fun test_softmax_negative_axis() {
        Utils.singleTestHelper(getTargetPath("test_softmax_negative_axis"))
    }
}
