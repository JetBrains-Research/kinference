package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "/shape/$dirName/"

    @Test
    fun test_shape() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape"))
    }

    @Test
    fun test_shape_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_example"))
    }
}
