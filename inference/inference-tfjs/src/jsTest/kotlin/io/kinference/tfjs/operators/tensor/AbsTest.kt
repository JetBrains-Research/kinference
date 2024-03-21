package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AbsTest {
    private fun getTargetPath(dirName: String) = "abs/$dirName/"

    @Test
    fun test_abs_default() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_abs"))
    }
}
