package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class FastGeluTest {
    private fun getTargetPath(dirName: String) = "/fastgelu/$dirName/"

    @Test
    fun test_fast_GELU_with_bias()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_fastgelu_with_bias"))
    }

    @Test
    fun test_fast_GELU_without_bias()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_fastgelu_without_bias"))
    }
}
