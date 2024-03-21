package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class FlattenTest {
    private fun getTargetPath(dirName: String) = "flatten/$dirName/"

    @Test
    fun test_flatten_axis_0() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_axis0"))
    }

    @Test
    fun test_flatten_axis_1() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_axis1"))
    }

    @Test
    fun test_flatten_axis_2() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_axis2"))
    }

    @Test
    fun test_flatten_axis_3() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_axis3"))
    }

    @Test
    fun test_flatten_default_axis() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_default_axis"))
    }

    @Test
    fun test_flatten_negative_axis_1() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_negative_axis1"))
    }

    @Test
    fun test_flatten_negative_axis_2() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_negative_axis2"))
    }

    @Test
    fun test_flatten_negative_axis_3() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_negative_axis3"))
    }

    @Test
    fun test_flatten_negative_axis_4() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_flatten_negative_axis4"))
    }
}
