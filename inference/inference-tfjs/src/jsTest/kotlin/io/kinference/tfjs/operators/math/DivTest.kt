package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class DivTest {
    private fun getTargetPath(dirName: String) = "div/$dirName/"

    @Test
    fun test_div() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_div"))
    }

    @Test
    fun test_div_broadcast() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_div_bcast"))
    }

    @Test
    fun test_div_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_div_example"))
    }

    @Test
    fun test_div_uint8() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_div_uint8"))
    }
}
