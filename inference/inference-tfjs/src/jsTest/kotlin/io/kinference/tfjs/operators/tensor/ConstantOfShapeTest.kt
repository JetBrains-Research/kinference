package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ConstantOfShapeTest {
    private fun getTargetPath(dirName: String) = "constant_of_shape/$dirName/"

    @Test
    fun test_ones()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_constantofshape_float_ones"))
    }

    @Test
    fun test_zero_shape()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_constantofshape_int_shape_zero"))
    }

    @Test
    fun test_zeros()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_constantofshape_int_zeros"))
    }
}
