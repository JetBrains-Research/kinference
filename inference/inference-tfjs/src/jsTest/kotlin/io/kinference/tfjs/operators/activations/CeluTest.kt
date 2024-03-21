package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CeluTest {
    private fun getTargetPath(dirName: String) = "celu/$dirName/"

    @Test
    fun test_celu() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_celu"))
    }

    @Test
    fun test_celu_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_celu_expanded"))
    }
}
