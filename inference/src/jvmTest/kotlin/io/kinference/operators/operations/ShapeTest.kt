package io.kinference.operators.operations

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "/shape/$dirName/"

    @Test
    fun `test shape`() {
        TestRunner.runFromResources(getTargetPath("test_shape"))
    }

    @Test
    fun `test shape example`() {
        TestRunner.runFromResources(getTargetPath("test_shape_example"))
    }
}
