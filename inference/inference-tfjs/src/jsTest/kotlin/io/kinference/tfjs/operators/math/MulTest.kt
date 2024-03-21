package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MulTest {
    private fun getTargetPath(dirName: String) = "mul/$dirName/"

    @Test
    fun test_mul()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mul"))
    }

    @Test
    fun test_mul_with_broadcast()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mul_bcast"))
    }

    @Test
    fun test_mul_defaults()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mul_example"))
    }
}
