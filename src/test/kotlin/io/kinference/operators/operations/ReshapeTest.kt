package io.kinference.operators.operations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class ReshapeTest {
    private fun getTargetPath(dirName: String) = "/reshape/$dirName/"

    @Test
    fun `test reshape with extended dimensions`() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_extended_dims"))
    }

    @Test
    fun `test reshape with negative dimension`() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_negative_dim"))
    }

    @Test
    fun `test reshape with negative extended dimensions`() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_negative_extended_dims"))
    }

    @Test
    fun `test reshape with one dimension`() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_one_dim"))
    }

    @Test
    fun `test reshape with reduced dimensions`() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_reduced_dims"))
    }

    @Test
    fun `test reshape with all reordered dimensions`() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_reordered_all_dims"))
    }

    @Test
    fun `test reshape with reordered last dimensions`() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_reordered_last_dims"))
    }

    @Test
    fun `test reshape with zero and negative dimension`() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_zero_and_negative_dim"))
    }

    @Test
    fun test_reshape_zero_dim() {
        Utils.tensorTestRunner(getTargetPath("test_reshape_zero_dim"))
    }
}
