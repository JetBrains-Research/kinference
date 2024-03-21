package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TanhTest {
    private fun getTargetPath(dirName: String) = "tanh/$dirName/"

    @Test
    fun test_tanh_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tanh_example"))
    }

    @Test
    fun test_tanh() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tanh"))
    }

    @Test
    fun test_tanh_scalar() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tanh_scalar"))
    }
}
