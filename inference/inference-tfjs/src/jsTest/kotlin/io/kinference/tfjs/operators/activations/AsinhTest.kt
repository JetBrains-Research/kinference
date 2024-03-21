package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AsinhTest {
    private fun getTargetPath(dirName: String) = "asinh/$dirName/"

    @Test
    fun test_asinh() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_asinh"))
    }

    @Test
    fun test_asinh_example() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_asinh_example"))
    }
}
