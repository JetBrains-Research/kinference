package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CeilTest {
    private fun getTargetPath(dirName: String) = "ceil/$dirName/"

    @Test
    fun test_ceil() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_ceil"))
    }

    @Test
    fun test_ceil_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_ceil_example"))
    }
}
