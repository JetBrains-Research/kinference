package io.kinference.tfjs.operators.reduce

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class ReduceMaxVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_max/v18/$dirName/"

    @Test
    fun test_reduce_max_default_axes_keepdim_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_default_axes_keepdim_example"))
    }

    @Test
    fun test_reduce_max_default_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_max_do_not_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_max_do_not_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_max_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_keepdims_example"))
    }

    @Test
    fun test_reduce_max_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_keepdims_random"))
    }

    @Test
    fun test_reduce_max_negative_axes_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_max_negative_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_negative_axes_keepdims_random"))
    }
}

class ReduceMaxVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_max/v1/$dirName/"

    @Test
    fun test_reduce_max_default_axes_keepdim_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_default_axes_keepdim_example"))
    }

    @Test
    fun test_reduce_max_default_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_max_do_not_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_max_do_not_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_max_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_keepdims_example"))
    }

    @Test
    fun test_reduce_max_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_keepdims_random"))
    }

    @Test
    fun test_reduce_max_negative_axes_keepdims_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_max_negative_axes_keepdims_random()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_max_negative_axes_keepdims_random"))
    }
}
