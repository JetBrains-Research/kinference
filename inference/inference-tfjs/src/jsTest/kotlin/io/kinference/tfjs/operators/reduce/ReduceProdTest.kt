package io.kinference.tfjs.operators.reduce

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class ReduceProdVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_prod/v18/$dirName/"

    @Test
    fun test_reduce_prod_default_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_default_axes_keepdims_example"))
    }

    /*
     * This test for JS requires delta = 2e-3 because JS doesn't have float32 type
     * and calculates everything with float64 which leads to bigger error
     */
    @Test
    fun test_reduce_prod_default_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_default_axes_keepdims_random"), delta = 2e-3)
    }

    @Test
    fun test_reduce_prod_do_not_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_do_not_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_negative_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_negative_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_negative_axes_keepdims_random"))
    }
}

class ReduceProdVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_prod/v1/$dirName/"

    @Test
    fun test_reduce_prod_default_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_default_axes_keepdims_example"))
    }

    /*
     * This test for JS requires delta = 2e-3 because JS doesn't have float32 type
     * and calculates everything with float64 which leads to bigger error
     */
    @Test
    fun test_reduce_prod_default_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_default_axes_keepdims_random"), delta = 2e-3)
    }

    @Test
    fun test_reduce_prod_do_not_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_do_not_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_negative_axes_keepdims_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_negative_axes_keepdims_random()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_negative_axes_keepdims_random"))
    }
}
