package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class HardmaxTest {
    private fun getTargetPath(dirName: String) = "hardmax/$dirName/"

    @Test
    fun test_hardmax_axis_0() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_0"))
    }

    @Test
    fun test_hardmax_axis_1() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_1"))
    }

    @Test
    fun test_hardmax_axis_2() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_axis_2"))
    }

    @Test
    fun test_hardmax_default_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_default_axis"))
    }

    @Test
    fun test_hardmax_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_example"))
    }

    @Test
    fun test_hardmax_negative_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_negative_axis"))
    }

    @Test
    fun test_hardmax_one_hot() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_hardmax_one_hot"))
    }
}
