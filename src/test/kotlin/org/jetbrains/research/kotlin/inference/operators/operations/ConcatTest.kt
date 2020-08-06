package org.jetbrains.research.kotlin.inference.operators.operations

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class ConcatTest {
    private fun getTargetPath(dirName: String) = "/concat/$dirName/"

    @Test
    fun `test concat 1D (axis=0)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_1d_axis_0"))
    }

    @Test
    fun `test concat 1D (axis=-1)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_1d_axis_negative_1"))
    }

    @Test
    fun `test concat 2D (axis=0)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_2d_axis_0"))
    }

    @Test
    fun `test concat 2D (axis=1)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_2d_axis_1"))
    }

    @Test
    fun `test concat 2D (axis=-1)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_2d_axis_negative_1"))
    }

    @Test
    fun `test concat 2D (axis=-2)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_2d_axis_negative_2"))
    }

    @Test
    fun `test concat 3D (axis=0)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_3d_axis_0"))
    }

    @Test
    fun `test concat 3D (axis=1)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_3d_axis_1"))
    }

    @Test
    fun `test concat 3D (axis=2)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_3d_axis_2"))
    }

    @Test
    fun `test concat 3D (axis=-1)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_2d_axis_negative_1"))
    }

    @Test
    fun `test concat 3D (axis=-2)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_2d_axis_negative_2"))
    }

    @Test
    fun `test concat 3D (axis=-3)`() {
        Utils.tensorTestRunner(getTargetPath("test_concat_3d_axis_negative_3"))
    }
}
