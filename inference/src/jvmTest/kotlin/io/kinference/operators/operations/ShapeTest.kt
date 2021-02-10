package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "/shape/$dirName/"

    @Test
    fun `test shape`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_shape"))
    }

    @Test
    fun `test shape example`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_shape_example"))
    }
}
