package io.kinference.webgpu.operators.tensor

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class FlattenTest {
    private fun getTargetPath(dirName: String) = "/flatten/$dirName/"

    @Test
    fun test_flatten_axis_0() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_axis0"))
    }

    @Test
    fun test_flatten_axis_1() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_axis1"))
    }

    @Test
    fun test_flatten_axis_2() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_axis2"))
    }

    @Test
    fun test_flatten_axis_3() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_axis3"))
    }

    @Test
    fun test_flatten_default_axis() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_default_axis"))
    }

    @Test
    fun test_flatten_negative_axis_1() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_negative_axis1"))
    }

    @Test
    fun test_flatten_negative_axis_2() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_negative_axis2"))
    }

    @Test
    fun test_flatten_negative_axis_3() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_negative_axis3"))
    }

    @Test
    fun test_flatten_negative_axis_4() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_flatten_negative_axis4"))
    }
}
