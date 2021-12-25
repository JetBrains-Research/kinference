package io.kinference.webgpu.operators.tensor

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class ConstantOfShapeTest {
    private fun getTargetPath(dirName: String) = "/constant_of_shape/$dirName/"

    @Test
    fun test_ones() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_constantofshape_float_ones"))
    }

    @Test
    fun test_zero_shape() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_constantofshape_int_shape_zero"))
    }

    @Test
    fun test_zeros() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_constantofshape_int_zeros"))
    }
}
