package io.kinference.tfjs.operators.reduce

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class ReduceSumTestVer1 {
    private fun getTargetPath(dirName: String) = "reduce_sum/v1/$dirName/"

    @Test
    fun test_reduce_sum_default_axes_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_default_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_do_not_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_do_not_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_negative_axes_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_negative_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_negative_axes_keepdims_random"))
    }
}

class ReduceSumTestV13 {
    private fun getTargetPath(dirName: String) = "reduce_sum/v13/$dirName/"

    @Test
    fun test_reduce_sum_default_axes_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_default_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_do_not_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_do_not_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_empty_axes_input_noop_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_empty_axes_input_noop_example"))
    }

    @Test
    fun test_reduce_sum_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_keepdims_random"))
    }

    @Test
    fun test_reduce_sum_negative_axes_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_sum_negative_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_sum_negative_axes_keepdims_random"))
    }
}
