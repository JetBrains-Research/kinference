package io.kinference.webgpu.operators.tensor

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class SqueezeTest {
    private fun getTargetPath(dirName: String) = "/squeeze/$dirName/"

    @Test
    fun test_squeeze() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_squeeze"))
    }

    @Test
    fun test_squeeze_with_negative_axes() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_squeeze_negative_axes"))
    }
}
