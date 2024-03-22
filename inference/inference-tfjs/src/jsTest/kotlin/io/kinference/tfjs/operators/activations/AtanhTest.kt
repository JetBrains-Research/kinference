package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AtanhTest {
    private fun getTargetPath(dirName: String) = "atanh/$dirName/"

    @Test
    fun test_atanh() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_atanh"))
    }

    @Test
    fun test_atanh_example() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_atanh_example"))
    }
}
