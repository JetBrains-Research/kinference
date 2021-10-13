package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GatherTest {
    private fun getTargetPath(dirName: String) = "/gather/$dirName/"

    @Test
    fun test_gather_0()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_0"))
    }

    @Test
    fun test_gather_1()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_1"))
    }

    @Test
    fun test_gather_with_negative_indices()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_negative_indices"))
    }
}
