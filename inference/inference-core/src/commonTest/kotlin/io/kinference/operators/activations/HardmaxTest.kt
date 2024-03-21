package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class HardmaxTest {
    private fun getTargetPath(dirName: String) = "hardmax/$dirName/"

    @Test
    fun test_hardmax_axis_0() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_0"))
    }

    @Test
    fun test_hardmax_axis_1() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_1"))
    }

    @Test
    fun test_hardmax_axis_2() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_2"))
    }

    @Test
    fun test_hardmax_default_axis() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_hardmax_default_axis"))
    }

    @Test
    fun test_hardmax_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_hardmax_example"))
    }

    @Test
    fun test_hardmax_negative_axis() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_hardmax_negative_axis"))
    }

    @Test
    fun test_hardmax_one_hot() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_hardmax_one_hot"))
    }
}
