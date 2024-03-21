package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class FastGeluTest {
    private fun getTargetPath(dirName: String) = "fastgelu/$dirName/"

    @Test
    fun test_fast_GELU_with_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_fastgelu_with_bias"))
    }

    @Test
    fun test_fast_GELU_without_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_fastgelu_without_bias"))
    }
}
