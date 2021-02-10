package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class FastGeluTest {
    private fun getTargetPath(dirName: String) = "/fastgelu/$dirName/"

    @Test
    fun `test fast GELU with bias`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_fastgelu_with_bias"))
    }

    @Test
    fun `test fast GELU without bias`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_fastgelu_without_bias"))
    }
}
