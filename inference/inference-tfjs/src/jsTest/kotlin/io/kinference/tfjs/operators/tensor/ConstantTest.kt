package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ConstantTest {
    private fun getTargetPath(dirName: String) = "constant/$dirName/"

    @Test
    fun test_constant() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_constant"))
    }

    @Test
    fun test_scalar_constant() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_scalar_constant"))
    }
}
