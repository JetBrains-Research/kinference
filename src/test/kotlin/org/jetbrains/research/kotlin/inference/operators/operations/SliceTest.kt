package org.jetbrains.research.kotlin.inference.operators.operations

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class SliceTest {
    private fun getTargetPath(dirName: String) = "/slice/$dirName/"

    @Test
    fun `test slice`() {
        Utils.tensorTestRunner(getTargetPath("test_slice"))
    }

    @Test
    fun `test slice with default axes`() {
        Utils.tensorTestRunner(getTargetPath("test_slice_default_axes"))
    }

    @Test
    fun `test slice with default steps`() {
        Utils.tensorTestRunner(getTargetPath("test_slice_default_steps"))
    }

    @Test
    fun `test slice end out of bounds`() {
        Utils.tensorTestRunner(getTargetPath("test_slice_end_out_of_bounds"))
    }

    @Test
    fun `test slice with negative index`() {
        Utils.tensorTestRunner(getTargetPath("test_slice_neg"))
    }

    @Test
    fun `test slice with negative steps`() {
        Utils.tensorTestRunner(getTargetPath("test_slice_neg_steps"))
    }

    @Test
    fun `test slice with negative axes`() {
        Utils.tensorTestRunner(getTargetPath("test_slice_negative_axes"))
    }

    @Test
    fun `test slice start out of bounds`() {
        Utils.tensorTestRunner(getTargetPath("test_slice_start_out_of_bounds"))
    }
}
