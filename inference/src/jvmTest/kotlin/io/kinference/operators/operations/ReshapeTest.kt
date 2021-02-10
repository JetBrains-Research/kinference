package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class ReshapeTest {
    private fun getTargetPath(dirName: String) = "/reshape/$dirName/"

    @Test
    fun `test reshape with extended dimensions`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_extended_dims"))
    }

    @Test
    fun `test reshape with negative dimension`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_negative_dim"))
    }

    @Test
    fun `test reshape with negative extended dimensions`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_negative_extended_dims"))
    }

    @Test
    fun `test reshape with one dimension`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_one_dim"))
    }

    @Test
    fun `test reshape with reduced dimensions`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_reduced_dims"))
    }

    @Test
    fun `test reshape with all reordered dimensions`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_reordered_all_dims"))
    }

    @Test
    fun `test reshape with reordered last dimensions`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_reordered_last_dims"))
    }

    @Test
    fun `test reshape with zero and negative dimension`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_zero_and_negative_dim"))
    }

    @Test
    fun test_reshape_zero_dim()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reshape_zero_dim"))
    }
}
