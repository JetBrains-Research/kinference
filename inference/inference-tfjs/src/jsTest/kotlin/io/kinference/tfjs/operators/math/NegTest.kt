package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class NegTest {
    private fun getTargetPath(dirName: String) = "neg/$dirName/"

    @Test
    fun test_neg() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_neg"))
    }

    @Test
    fun test_neg_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_neg_example"))
    }
}
