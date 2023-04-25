package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AcosTest {
    private fun getTargetPath(dirName: String) = "acos/$dirName/"

    @Test
    fun test_acos() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_acos"))
    }

    @Test
    fun test_acos_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_acos_example"))
    }
}
