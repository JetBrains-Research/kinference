package io.kinference.webgpu.operators.math

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class DivTest {
    private fun getTargetPath(dirName: String) = "/div/$dirName/"

    @Test
    fun test_div() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_div"))
    }

    @Test
    fun test_div_broadcast() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_div_bcast"))
    }

    @Test
    fun test_div_example() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_div_example"))
    }

    @Test
    fun test_div_uint8() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_div_uint8"))
    }
}
