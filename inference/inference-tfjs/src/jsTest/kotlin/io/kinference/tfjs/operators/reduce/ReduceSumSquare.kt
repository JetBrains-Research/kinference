package io.kinference.tfjs.operators.reduce

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ReduceSumSquareVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_sum_square/v18/$dirName/"

    @Test
    fun test_reduce_sum_square_default_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_square_default_axes_keepdims_example_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_default_axes_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_sum_square_default_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_square_default_axes_keepdims_random_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_default_axes_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_sum_square_do_not_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_square_do_not_keepdims_example_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_do_not_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_sum_square_do_not_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_square_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_square_keepdims_example_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_sum_square_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_square_keepdims_random_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_sum_square_negative_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_square_negative_axes_keepdims_example_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_negative_axes_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_sum_square_negative_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_negative_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_square_negative_axes_keepdims_random_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_negative_axes_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_sum_square_do_not_keepdims_random_expanded()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_do_not_keepdims_random_expanded"))
    }
}

class ReduceSumSquareVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_sum_square/v1/$dirName/"

    @Test
    fun test_reduce_sum_square_default_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_square_default_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_square_do_not_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_square_do_not_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_square_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_square_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_square_negative_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_square_negative_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_square_negative_axes_keepdims_random"))
    }
}
