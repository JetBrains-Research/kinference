package io.kinference.operators.reduce

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class ReduceL1Ver18Test {
    private fun getTargetPath(dirName: String) = "reduce_l1/v18/$dirName/"

    @Test
    fun test_reduce_l1_default_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_l1_default_axes_keepdims_example_expanded()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_l1_default_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_l1_default_axes_keepdims_random_expanded()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_example_expanded()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_l1_keep_dims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_example"))
    }

    @Test
    fun test_reduce_l1_keep_dims_example_expanded()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_example_expanded"))
    }

    @Test
    fun test_reduce_l1_keep_dims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_random"))
    }

    @Test
    fun test_reduce_l1_keep_dims_random_expanded()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_random_expanded"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_example"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_example_expanded()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_example_expanded"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_random"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_random_expanded()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_random_expanded"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_random_expanded()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_random_expanded"))
    }
}

class ReduceL1Ver1Test {
    private fun getTargetPath(dirName: String) = "reduce_l1/v1/$dirName/"

    @Test
    fun test_reduce_l1_default_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_l1_default_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_l1_do_not_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_l1_keep_dims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_example"))
    }

    @Test
    fun test_reduce_l1_keep_dims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_keep_dims_random"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_example"))
    }

    @Test
    fun test_reduce_l1_negative_axes_keep_dims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_l1_negative_axes_keep_dims_random"))
    }
}
