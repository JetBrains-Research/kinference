package io.kinference.operators.operations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "/shape/$dirName/"

    @Test
    fun `test shape`() {
        Utils.tensorTestRunner(getTargetPath("test_shape"))
    }

    @Test
    fun `test shape example`() {
        Utils.tensorTestRunner(getTargetPath("test_shape_example"))
    }
}
