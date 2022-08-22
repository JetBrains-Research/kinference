package io.kinference.tfjs.operators.logical

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LessTest {
    private fun getTargetPath(dirName: String) = "less/$dirName/"

    @Test
    fun test_less() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_less"))
    }

    @Test
    fun test_less_with_broadcast() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_less_bcast"))
    }
}
