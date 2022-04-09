package io.kinference.webgpu.operators.math

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "/matmul/$dirName/"

    @Test
    fun test_matmul_2D() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun test_matmul_3D() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun test_matmul_4D() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_matmul_4d"))
    }

    @Test
    fun test_matmul_packed_vec4() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_matmul_packed_vec4"))
    }

    @Test
    fun test_matmul_packed_unaligned() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_matmul_packed_unaligned"))
    }
}
