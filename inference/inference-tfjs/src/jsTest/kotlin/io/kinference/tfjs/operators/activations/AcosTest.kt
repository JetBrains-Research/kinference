package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AcosTest {
    private fun getTargetPath(dirName: String) = "acos/$dirName/"

    @Test
    fun test_acos() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_acos"))
    }

    @Test
    fun test_acos_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_acos_example"))
    }
}
