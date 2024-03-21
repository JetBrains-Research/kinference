package io.kinference.tfjs.operators.reduce

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class ReduceMeanVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_mean/v18/$dirName/"

    @Test
    fun test_reduce_mean_default_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_default_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_do_not_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_do_not_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_negative_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_negative_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_negative_axes_keepdims_random"))
    }
}

class ReduceMeanVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_mean/v1/$dirName/"

    @Test
    fun test_reduce_mean_default_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_default_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_do_not_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_do_not_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_keepdims_random"))
    }

    @Test
    fun test_reduce_mean_negative_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_mean_negative_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_mean_negative_axes_keepdims_random"))
    }
}
