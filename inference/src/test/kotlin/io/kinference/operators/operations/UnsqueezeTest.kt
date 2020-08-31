package io.kinference.operators.operations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class UnsqueezeTest {
    private fun getTargetPath(dirName: String) = "/unsqueeze/$dirName/"

    @Test
    fun `test unsqueeze axis 0`() {
        Utils.tensorTestRunner(getTargetPath("test_unsqueeze_axis_0"))
    }

    @Test
    fun `test unsqueeze axis 1`() {
        Utils.tensorTestRunner(getTargetPath("test_unsqueeze_axis_1"))
    }

    @Test
    fun `test unsqueeze axis 2`() {
        Utils.tensorTestRunner(getTargetPath("test_unsqueeze_axis_2"))
    }

    @Test
    fun `test unsqueeze axis 3`() {
        Utils.tensorTestRunner(getTargetPath("test_unsqueeze_axis_3"))
    }

    @Test
    fun `test unsqueeze with negative axes`() {
        Utils.tensorTestRunner(getTargetPath("test_unsqueeze_negative_axes"))
    }

    @Test
    fun `test unsqueeze three axes`() {
        Utils.tensorTestRunner(getTargetPath("test_unsqueeze_three_axes"))
    }

    @Test
    fun `test unsqueeze two axes`() {
        Utils.tensorTestRunner(getTargetPath("test_unsqueeze_two_axes"))
    }

    @Test
    fun `test unsqueeze unsorted axes`() {
        Utils.tensorTestRunner(getTargetPath("test_unsqueeze_unsorted_axes"))
    }
}
