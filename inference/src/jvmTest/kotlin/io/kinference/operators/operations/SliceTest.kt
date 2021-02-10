package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class SliceTest {
    private fun getTargetPath(dirName: String) = "/slice/$dirName/"

    @Test
    fun `test slice`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_slice"))
    }

    @Test
    fun `test slice with default axes`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_slice_default_axes"))
    }

    @Test
    fun `test slice with default steps`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_slice_default_steps"))
    }

    @Test
    fun `test slice end out of bounds`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_slice_end_out_of_bounds"))
    }

    @Test
    fun `test slice with negative index`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_slice_neg"))
    }

    @Test
    fun `test slice with negative steps`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_slice_neg_steps"))
    }

    @Test
    fun `test slice with negative axes`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_slice_negative_axes"))
    }

    @Test
    fun `test slice start out of bounds`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_slice_start_out_of_bounds"))
    }
}
