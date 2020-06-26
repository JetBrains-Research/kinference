package org.jetbrains.research.kotlin.mpp.inference.operators.operations

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class ConcatTest {
    private fun getTargetPath(dirName: String) = "/concat/$dirName/"

    @Test
    fun test_concat_1d_axis_0() {
        Utils.singleTestHelper(getTargetPath("test_concat_1d_axis_0"))
    }

    @Test
    fun test_concat_1d_axis_negative_1() {
        Utils.singleTestHelper(getTargetPath("test_concat_1d_axis_negative_1"))
    }

    @Test
    fun test_concat_2d_axis_0() {
        Utils.singleTestHelper(getTargetPath("test_concat_2d_axis_0"))
    }

    @Test
    fun test_concat_2d_axis_1() {
        Utils.singleTestHelper(getTargetPath("test_concat_2d_axis_1"))
    }

    @Test
    fun test_concat_2d_axis_negative_1() {
        Utils.singleTestHelper(getTargetPath("test_concat_2d_axis_negative_1"))
    }

    @Test
    fun test_concat_2d_axis_negative_2() {
        Utils.singleTestHelper(getTargetPath("test_concat_2d_axis_negative_2"))
    }

    @Test
    fun test_concat_3d_axis_0() {
        Utils.singleTestHelper(getTargetPath("test_concat_3d_axis_0"))
    }

    @Test
    fun test_concat_3d_axis_1() {
        Utils.singleTestHelper(getTargetPath("test_concat_3d_axis_1"))
    }

    @Test
    fun test_concat_3d_axis_2() {
        Utils.singleTestHelper(getTargetPath("test_concat_3d_axis_2"))
    }

    @Test
    fun test_concat_3d_axis_negative_1() {
        Utils.singleTestHelper(getTargetPath("test_concat_2d_axis_negative_1"))
    }

    @Test
    fun test_concat_3d_axis_negative_2() {
        Utils.singleTestHelper(getTargetPath("test_concat_2d_axis_negative_2"))
    }

    @Test
    fun test_concat_3d_axis_negative_3() {
        Utils.singleTestHelper(getTargetPath("test_concat_3d_axis_negative_3"))
    }
}
