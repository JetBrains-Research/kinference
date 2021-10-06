package io.kinference.operators.tensor

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class UnsqueezeTest {
    private fun getTargetPath(dirName: String) = "/unsqueeze/$dirName/"

    @Test
    fun test_unsqueeze_axis_0()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_0"))
    }

    @Test
    fun test_unsqueeze_axis_1()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_1"))
    }

    @Test
    fun test_unsqueeze_axis_2()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_2"))
    }

    @Test
    fun test_unsqueeze_axis_3()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_3"))
    }

    @Test
    fun test_unsqueeze_with_negative_axes()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_negative_axes"))
    }

    @Test
    fun test_unsqueeze_three_axes()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_three_axes"))
    }

    @Test
    fun test_unsqueeze_two_axes()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_two_axes"))
    }

    @Test
    fun test_unsqueeze_unsorted_axes()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_unsorted_axes"))
    }
}
