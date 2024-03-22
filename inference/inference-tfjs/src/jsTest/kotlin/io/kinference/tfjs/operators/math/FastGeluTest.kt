package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class FastGeluTest {
    private fun getTargetPath(dirName: String) = "fastgelu/$dirName/"

    @Test
    fun test_fast_GELU_with_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_fastgelu_with_bias"))
    }

    @Test
    fun test_fast_GELU_without_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_fastgelu_without_bias"))
    }
}
