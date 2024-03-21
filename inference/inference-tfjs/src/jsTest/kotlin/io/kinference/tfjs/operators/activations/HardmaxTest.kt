package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class HardmaxTest {
    private fun getTargetPath(dirName: String) = "hardmax/$dirName/"

    @Test
    fun test_hardmax_axis_0() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_0"))
    }

    @Test
    fun test_hardmax_axis_1() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_1"))
    }

    @Test
    fun test_hardmax_axis_2() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_2"))
    }

    @Test
    fun test_hardmax_default_axis() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_default_axis"))
    }

    @Test
    fun test_hardmax_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_example"))
    }

    @Test
    fun test_hardmax_negative_axis() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_negative_axis"))
    }

    @Test
    fun test_hardmax_one_hot() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_one_hot"))
    }
}
