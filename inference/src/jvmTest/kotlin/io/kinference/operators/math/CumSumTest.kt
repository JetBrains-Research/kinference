package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class CumSumTest {
    private fun getTargetPath(dirName: String) = "/cumsum/$dirName/"

    @Test
    fun `test cumulative sum for 1d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_1d"))
    }

    @Test
    fun `test exclusive cumulative sum for 1d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_1d_exclusive"))
    }

    @Test
    fun `test reverse exclusive cumulative sum for 1d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_1d_reverse_exclusive"))
    }

    @Test
    fun `test cumulative sum along axis=0 for 2d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_axis_0"))
    }

    @Test
    fun `test cumulative sum along axis=1 for 2d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_axis_1"))
    }

    @Test
    fun `test cumulative sum along negative axis for 2d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_negative_axis"))
    }

    @Test
    fun `test reverse exclusive cumulative sum along axis=1 for 2d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_axis_1_reverse_exclusive"))
    }

    @Test
    fun `test reverse cumulative sum along axis=0 for 2d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_axis_0_reverse"))
    }

    @Test
    fun `test reverse cumulative sum along axis=1 for 3d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_3d_axis_1_reverse"))
    }

    @Test
    fun `test reverse exclusive cumulative sum along negative axis for 3d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_3d_negative_axis_reverse_exclusive"))
    }

    @Test
    fun `test cumulative sum along axis=2 for 4d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_4d_axis_2"))
    }

    @Test
    fun `test exclusive cumulative sum along axis=0 for 4d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_cumsum_4d_axis_0_exclusive"))
    }
}
