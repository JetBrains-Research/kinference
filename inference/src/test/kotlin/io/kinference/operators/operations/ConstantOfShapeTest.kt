package io.kinference.operators.operations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class ConstantOfShapeTest {
    private fun getTargetPath(dirName: String) = "/constant_of_shape/$dirName/"

    @Test
    fun `test ones`() {
        Utils.tensorTestRunner(getTargetPath("test_constantofshape_float_ones"))
    }

    @Test
    fun `test zero shape`() {
        Utils.tensorTestRunner(getTargetPath("test_constantofshape_int_shape_zero"))
    }

    @Test
    fun `test zeros`() {
        Utils.tensorTestRunner(getTargetPath("test_constantofshape_int_zeros"))
    }
}
