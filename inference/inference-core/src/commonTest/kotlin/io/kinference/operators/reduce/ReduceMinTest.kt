package io.kinference.operators.reduce

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class ReduceMinVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_min/v18/$dirName/"

    @Test
    fun test_reduce_min_default_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_min_default_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_min_do_not_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_min_do_not_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_min_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_keepdims_example"))
    }

    @Test
    fun test_reduce_min_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_keepdims_random"))
    }

    @Test
    fun test_reduce_min_negative_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_min_negative_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_negative_axes_keepdims_random"))
    }
}

class ReduceMinVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_min/v1/$dirName/"

    @Test
    fun test_reduce_min_default_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_min_default_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_min_do_not_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_min_do_not_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_min_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_keepdims_example"))
    }

    @Test
    fun test_reduce_min_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_keepdims_random"))
    }

    @Test
    fun test_reduce_min_negative_axes_keepdims_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_min_negative_axes_keepdims_random()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_min_negative_axes_keepdims_random"))
    }
}
