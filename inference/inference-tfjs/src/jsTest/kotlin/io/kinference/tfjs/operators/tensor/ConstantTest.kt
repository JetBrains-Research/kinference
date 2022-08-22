package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ConstantTest {
    private fun getTargetPath(dirName: String) = "constant/$dirName/"

    @Test
    fun test_constant() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_constant"))
    }

    @Test
    fun test_scalar_constant() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_scalar_constant"))
    }
}
