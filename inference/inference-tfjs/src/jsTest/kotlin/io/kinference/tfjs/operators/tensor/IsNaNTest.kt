package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IsNaNTest {
    private fun getTargetPath(dirName: String) = "isnan/$dirName/"

    @Test
    fun test_isnan() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_isnan"))
    }
}
