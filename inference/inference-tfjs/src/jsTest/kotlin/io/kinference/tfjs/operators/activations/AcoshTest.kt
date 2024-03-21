package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AcoshTest {
    private fun getTargetPath(dirName: String) = "acosh/$dirName/"

    @Test
    fun test_acosh() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_acosh"))
    }

    @Test
    fun test_acosh_example() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_acosh_example"))
    }
}
