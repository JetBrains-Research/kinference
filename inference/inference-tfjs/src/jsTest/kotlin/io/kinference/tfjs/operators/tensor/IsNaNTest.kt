package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class IsNaNTest {
    private fun getTargetPath(dirName: String) = "isnan/$dirName/"

    @Test
    fun test_isnan() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_isnan"))
    }
}
