package io.kinference.operators.reduce

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class ReduceMeanVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_mean/v18/$dirName/"

    @Test
    fun test_reduce_mean_default_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_default_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_do_not_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_do_not_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_negative_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_negative_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_negative_axes_keepdims_random"))
    }
}

class ReduceMeanVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_mean/v1/$dirName/"

    @Test
    fun test_reduce_mean_default_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_default_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_do_not_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_do_not_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_negative_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_negative_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_negative_axes_keepdims_random"))
    }
}
