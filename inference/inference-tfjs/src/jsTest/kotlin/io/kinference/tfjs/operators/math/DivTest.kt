package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class DivTest {
    private fun getTargetPath(dirName: String) = "div/$dirName/"

    @Test
    fun test_div() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_div"))
    }

    @Test
    fun test_div_broadcast() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_div_bcast"))
    }

    @Test
    fun test_div_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_div_example"))
    }

    @Test
    fun test_div_uint8() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_div_uint8"))
    }
}
