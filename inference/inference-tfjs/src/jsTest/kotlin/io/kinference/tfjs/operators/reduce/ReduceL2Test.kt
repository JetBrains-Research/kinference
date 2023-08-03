package io.kinference.tfjs.operators.reduce

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class ReduceL2Ver18Test {
    private fun getTargetPath(dirName: String) = "reduce_l2/v18/$dirName/"

    @Test
    fun test_reduce_l2_default_axes_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_l2_default_axes_keepdims_example_expanded()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_default_axes_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_l2_default_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_l2_default_axes_keepdims_random_expanded()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_default_axes_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_l2_do_not_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_l2_do_not_keepdims_example_expanded()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_do_not_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_l2_do_not_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_l2_do_not_keepdims_random_expanded()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_do_not_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_l2_keep_dims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_keep_dims_example"))
    }

    @Test
    fun test_reduce_l2_keep_dims_example_expanded()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_keep_dims_example_expanded"))
    }

    @Test
    fun test_reduce_l2_keep_dims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_keep_dims_random"))
    }

    @Test
    fun test_reduce_l2_keep_dims_random_expanded()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_keep_dims_random_expanded"))
    }

    @Test
    fun test_reduce_l2_negative_axes_keep_dims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_negative_axes_keep_dims_example"))
    }

    @Test
    fun test_reduce_l2_negative_axes_keep_dims_example_expanded()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_negative_axes_keep_dims_example_expanded"))
    }

    @Test
    fun test_reduce_l2_negative_axes_keep_dims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_negative_axes_keep_dims_random"))
    }

    @Test
    fun test_reduce_l2_negative_axes_keep_dims_random_expanded()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_negative_axes_keep_dims_random_expanded"))
    }
}

class ReduceL2Ver1Test {
    private fun getTargetPath(dirName: String) = "reduce_l2/v1/$dirName/"

    @Test
    fun test_reduce_l2_default_axes_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_l2_default_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_l2_do_not_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_l2_do_not_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_l2_keep_dims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_keep_dims_example"))
    }

    @Test
    fun test_reduce_l2_keep_dims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_keep_dims_random"))
    }

    @Test
    fun test_reduce_l2_negative_axes_keep_dims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_negative_axes_keep_dims_example"))
    }

    @Test
    fun test_reduce_l2_negative_axes_keep_dims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_l2_negative_axes_keep_dims_random"))
    }
}
