package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class NegTest {
    private fun getTargetPath(dirName: String) = "neg/$dirName/"

    @Test
    fun test_neg() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_neg"))
    }

    @Test
    fun test_neg_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_neg_example"))
    }
}
