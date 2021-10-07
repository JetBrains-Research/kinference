package io.kinference.operators.operations

import io.kinference.runners.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

class SliceTest {
    private fun getTargetPath(dirName: String) = "/slice/$dirName/"

    @Test
    fun test_slice() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_slice"))
    }

    @Test
    fun test_slice_with_default_axes() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_slice_default_axes"))
    }

    @Test
    fun test_slice_with_default_steps() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_slice_default_steps"))
    }

    @Test
    fun test_slice_end_out_of_bounds() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_slice_end_out_of_bounds"))
    }

    @Test
    fun test_slice_with_negative_index() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_slice_neg"))
    }

    @Test
    fun test_slice_with_negative_steps() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_slice_neg_steps"))
    }

    @Test
    fun test_slice_with_negative_axes() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_slice_negative_axes"))
    }

    @Test
    fun test_slice_start_out_of_bounds() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_slice_start_out_of_bounds"))
    }
}
