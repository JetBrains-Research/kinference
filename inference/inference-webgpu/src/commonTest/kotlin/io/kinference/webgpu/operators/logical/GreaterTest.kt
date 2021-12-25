package io.kinference.webgpu.operators.logical

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class GreaterTest {
    private fun getTargetPath(dirName: String) = "/greater/$dirName/"

    @Test
    fun test_greater() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_greater"))
    }

    @Test
    fun test_greater_with_broadcast() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_greater_bcast"))
    }
}
