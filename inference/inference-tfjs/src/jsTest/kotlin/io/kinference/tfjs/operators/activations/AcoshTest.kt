package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AcoshTest {
    private fun getTargetPath(dirName: String) = "acosh/$dirName/"

    @Test
    fun test_acosh() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_acosh"))
    }

    @Test
    fun test_acosh_example() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_acosh_example"))
    }
}
