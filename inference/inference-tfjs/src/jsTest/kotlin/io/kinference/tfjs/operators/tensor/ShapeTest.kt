package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "/shape/$dirName/"

    @Test
    fun test_shape()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape"))
    }

    @Test
    fun test_shape_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_example"))
    }
}
