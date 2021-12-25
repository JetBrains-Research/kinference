package io.kinference.webgpu.operators.logical

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class LessTest {
    private fun getTargetPath(dirName: String) = "/less/$dirName/"

    @Test
    fun test_less() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_less"))
    }

    @Test
    fun test_less_with_broadcast() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_less_bcast"))
    }
}
