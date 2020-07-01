package org.jetbrains.research.kotlin.mpp.inference.operators.operations

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class SplitTest {
    private fun getTargetPath(dirName: String) = "/split/$dirName/"

    @Test
    fun test_split_equal_parts_1d() {
        Utils.tensorTestRunner(getTargetPath("test_split_equal_parts_1d"))
    }

    @Test
    fun test_split_equal_parts_2d() {
        Utils.tensorTestRunner(getTargetPath("test_split_equal_parts_2d"))
    }

    @Test
    fun test_split_equal_parts_default_axis() {
        Utils.tensorTestRunner(getTargetPath("test_split_equal_parts_default_axis"))
    }

    @Test
    fun test_split_variable_parts_1d() {
        Utils.tensorTestRunner(getTargetPath("test_split_variable_parts_1d"))
    }

    @Test
    fun test_split_variable_parts_2d() {
        Utils.tensorTestRunner(getTargetPath("test_split_equal_parts_2d"))
    }

    @Test
    fun test_split_variable_parts_default_axis() {
        Utils.tensorTestRunner(getTargetPath("test_split_variable_parts_default_axis"))
    }

    @Test
    fun test_split_zero_size_splits() {
        Utils.tensorTestRunner(getTargetPath("test_split_zero_size_splits"))
    }
}
