package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "/shape/$dirName/"

    @Test
    fun test_shape()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_shape"))
    }

    @Test
    fun test_shape_example()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_shape_example"))
    }
}
