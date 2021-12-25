package io.kinference.webgpu.operators.logical

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class EqualTest {
    private fun getTargetPath(dirName: String) = "/equal/$dirName/"

    @Test
    fun test_equal() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_equal"))
    }

    @Test
    fun test_equal_with_broadcast() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_equal_bcast"))
    }
}
