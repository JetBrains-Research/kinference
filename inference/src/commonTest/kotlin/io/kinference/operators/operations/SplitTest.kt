package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SplitTest {
    private fun getTargetPath(dirName: String) = "/split/$dirName/"

    @Test
    fun test_split_into_equal_parts_1D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_split_equal_parts_1d"))
    }

    @Test
    fun test_split_into_equal_parts_2D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_split_equal_parts_2d"))
    }

    @Test
    fun test_split_into_equal_parts_default_axis()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_split_equal_parts_default_axis"))
    }

    @Test
    fun test_split_into_variable_parts_1D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_split_variable_parts_1d"))
    }

    @Test
    fun test_split_into_variable_parts_2D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_split_equal_parts_2d"))
    }

    @Test
    fun test_split_into_variable_parts_default_axis()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_split_variable_parts_default_axis"))
    }

    @Test
    fun test_split_into_zero_size_splits()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_split_zero_size_splits"))
    }
}
