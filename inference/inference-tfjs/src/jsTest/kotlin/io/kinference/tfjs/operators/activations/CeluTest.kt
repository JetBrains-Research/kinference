package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CeluTest {
    private fun getTargetPath(dirName: String) = "celu/$dirName/"

    @Test
    fun test_celu() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_celu"))
    }

    @Test
    fun test_celu_expanded() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_celu_expanded"))
    }
}
