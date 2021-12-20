package io.kinference.webgpu.operators.tensor

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class ConstantTest {
    private fun getTargetPath(dirName: String) = "/constant/$dirName/"

    @Test
    fun test_constant() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_constant"))
    }

    @Test
    fun test_scalar_constant() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_scalar_constant"))
    }
}
