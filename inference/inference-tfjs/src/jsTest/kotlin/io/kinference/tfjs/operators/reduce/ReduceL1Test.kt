package io.kinference.tfjs.operators.reduce

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ReduceL1Ver18Test {
    private fun getTargetPath(dirName: String) = "reduce_l1/v18/$dirName/"

    @Test
    fun test_reduce_l1_default_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_l1_default_axes_keepdims_example_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_l1_default_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_l1_default_axes_keepdims_random_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_example_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_l1_keep_dims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_example"))
    }

    @Test
    fun test_reduce_l1_keep_dims_example_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_example_expanded"))
    }

    @Test
    fun test_reduce_l1_keep_dims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_random"))
    }

    @Test
    fun test_reduce_l1_keep_dims_random_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_random_expanded"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_example"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_example_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_example_expanded"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_random"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_random_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_random_expanded"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_random_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_random_expanded"))
    }
}

class ReduceL1Ver1Test {
    private fun getTargetPath(dirName: String) = "reduce_l1/v1/$dirName/"

    @Test
    fun test_reduce_l1_default_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_l1_default_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_l1_keep_dims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_example"))
    }

    @Test
    fun test_reduce_l1_keep_dims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_random"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_example"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_random"))
    }
}
