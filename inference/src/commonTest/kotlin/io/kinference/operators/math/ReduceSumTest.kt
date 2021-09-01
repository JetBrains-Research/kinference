package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class ReduceSumTest {
    private fun getTargetPath(dirName: String) = "/reduce_sum/$dirName/"

    @Test
    fun test_reduce_sum_default_axes_keepdims_example()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_default_axes_keepdims_random()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_do_not_keepdims_example()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_do_not_keepdims_random()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_keepdims_example()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_keepdims_random()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_negative_axes_keepdims_example()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_negative_axes_keepdims_random()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_negative_axes_keepdims_random"))
    }
}
