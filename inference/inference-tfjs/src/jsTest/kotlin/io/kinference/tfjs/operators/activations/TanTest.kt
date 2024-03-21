package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TanTest {
    private fun getTargetPath(dirName: String) = "tan/$dirName/"

    @Test
    fun test_tan_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tan_example"))
    }

    @Test
    fun test_tan() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tan"))
    }
}
