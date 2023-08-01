package io.kinference.operators.reduce

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class ReduceProdVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_prod/v18/$dirName/"

    @Test
    fun test_reduce_prod_default_axes_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_default_axes_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_do_not_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_do_not_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_negative_axes_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_negative_axes_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_negative_axes_keepdims_random"))
    }
}

class ReduceProdVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_prod/v1/$dirName/"

    @Test
    fun test_reduce_prod_default_axes_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_default_axes_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_do_not_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_do_not_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_keepdims_random"))
    }

    @Test
    fun test_reduce_prod_negative_axes_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_prod_negative_axes_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_prod_negative_axes_keepdims_random"))
    }
}
