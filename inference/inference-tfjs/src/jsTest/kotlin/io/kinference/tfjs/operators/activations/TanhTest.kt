package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TanhTest {
    private fun getTargetPath(dirName: String) = "tanh/$dirName/"

    @Test
    fun test_tanh_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tanh_example"))
    }

    @Test
    fun test_tanh() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tanh"))
    }

    @Test
    fun test_tanh_scalar() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tanh_scalar"))
    }
}
