package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MulTest {
    private fun getTargetPath(dirName: String) = "mul/$dirName/"

    @Test
    fun test_mul()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mul"))
    }

    @Test
    fun test_mul_with_broadcast()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mul_bcast"))
    }

    @Test
    fun test_mul_defaults()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mul_example"))
    }
}
