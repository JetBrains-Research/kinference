package io.kinference.operators.reduce

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class ReduceLogSumVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_log_sum/v18/$dirName/"

    @Test
    fun test_reduce_log_sum_asc_axes()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_asc_axes"))
    }

    @Test
    fun test_reduce_log_sum_asc_axes_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_asc_axes_expanded"))
    }

    @Test
    fun test_reduce_log_sum_default()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_default"))
    }

    @Test
    fun test_reduce_log_sum_default_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_default_expanded"))
    }

    @Test
    fun test_reduce_log_sum_desc_axes()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_desc_axes"))
    }

    @Test
    fun test_reduce_log_sum_desc_axes_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_desc_axes_expanded"))
    }

    @Test
    fun test_reduce_log_sum_negative_axes()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_negative_axes"))
    }

    @Test
    fun test_reduce_log_sum_negative_axes_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_negative_axes_expanded"))
    }
}

class ReduceLogSumVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_log_sum/v1/$dirName/"

    @Test
    fun test_reduce_log_sum()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum"))
    }

    @Test
    fun test_reduce_log_sum_asc_axes()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_asc_axes"))
    }

    @Test
    fun test_reduce_log_sum_default()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_default"))
    }

    @Test
    fun test_reduce_log_sum_desc_axes()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_desc_axes"))
    }

    @Test
    fun test_reduce_log_sum_negative_axes()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_negative_axes"))
    }
}
