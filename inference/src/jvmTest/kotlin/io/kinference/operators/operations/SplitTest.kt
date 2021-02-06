package io.kinference.operators.operations

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class SplitTest {
    private fun getTargetPath(dirName: String) = "/split/$dirName/"

    @Test
    fun `test split into equal parts 1D`() {
        TestRunner.runFromResources(getTargetPath("test_split_equal_parts_1d"))
    }

    @Test
    fun `test split into equal parts 2D`() {
        TestRunner.runFromResources(getTargetPath("test_split_equal_parts_2d"))
    }

    @Test
    fun `test split into equal parts default axis`() {
        TestRunner.runFromResources(getTargetPath("test_split_equal_parts_default_axis"))
    }

    @Test
    fun `test split into variable parts 1D`() {
        TestRunner.runFromResources(getTargetPath("test_split_variable_parts_1d"))
    }

    @Test
    fun `test split into variable parts 2D`() {
        TestRunner.runFromResources(getTargetPath("test_split_equal_parts_2d"))
    }

    @Test
    fun `test split into variable parts default axis`() {
        TestRunner.runFromResources(getTargetPath("test_split_variable_parts_default_axis"))
    }

    @Test
    fun `test split into zero size splits`() {
        TestRunner.runFromResources(getTargetPath("test_split_zero_size_splits"))
    }
}
