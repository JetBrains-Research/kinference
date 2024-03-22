package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CeilTest {
    private fun getTargetPath(dirName: String) = "ceil/$dirName/"

    @Test
    fun test_ceil() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_ceil"))
    }

    @Test
    fun test_ceil_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_ceil_example"))
    }
}
