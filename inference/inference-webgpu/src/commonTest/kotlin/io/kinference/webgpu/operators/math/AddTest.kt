package io.kinference.webgpu.operators.math

import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "/add/$dirName/"

    @Test
    fun test_add() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_add"))
    }

    @Test
    fun test_add_broadcast() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_add_bcast"))
    }

    @Test
    fun test_add_scalar() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_add_scalar"))
    }
}
