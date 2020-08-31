package io.kinference.operators.activations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class SoftmaxTest {
    private fun getTargetPath(dirName: String) = "/softmax/$dirName/"

    @Test
    fun `test softmax (axis=0)`() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_axis_0"))
    }

    @Test
    fun `test softmax (axis=1)`() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_axis_1"))
    }

    @Test
    fun `test softmax (axis=2)`() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_axis_2"))
    }

    @Test
    fun `test softmax with default axis`() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_default_axis"))
    }

    @Test
    fun `test softmax with large number`() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_large_number"))
    }

    @Test
    fun `test softmax with negative axis`() {
        Utils.tensorTestRunner(getTargetPath("test_softmax_negative_axis"))
    }
}
