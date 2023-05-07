package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CumSumTest {
    private fun getTargetPath(dirName: String) = "cumsum/$dirName/"

    @Test
    fun test_cumulative_sum_for_1d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_1d"))
    }

    @Test
    fun test_exclusive_cumulative_sum_for_1d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_1d_exclusive"))
    }

    @Test
    fun test_reverse_exclusive_cumulative_sum_for_1d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_1d_reverse_exclusive"))
    }

    @Test
    fun test_cumulative_sum_along_axis_0_for_2d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_axis_0"))
    }

    @Test
    fun test_cumulative_sum_along_axis_1_for_2d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_axis_1"))
    }

    @Test
    fun test_cumulative_sum_along_negative_axis_for_2d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_negative_axis"))
    }

    @Test
    fun test_reverse_exclusive_cumulative_sum_along_axis_1_for_2d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_axis_1_reverse_exclusive"))
    }

    @Test
    fun test_reverse_cumulative_sum_along_axis_0_for_2d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_2d_axis_0_reverse"))
    }

    @Test
    fun test_reverse_cumulative_sum_along_axis_1_for_3d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_3d_axis_1_reverse"))
    }

    @Test
    fun test_reverse_exclusive_cumulative_sum_along_negative_axis_for_3d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_3d_negative_axis_reverse_exclusive"))
    }

    @Test
    fun test_cumulative_sum_along_axis_2_for_4d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_4d_axis_2"))
    }

    @Test
    fun test_exclusive_cumulative_sum_along_axis_0_for_4d_data() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_cumsum_4d_axis_0_exclusive"))
    }
}
