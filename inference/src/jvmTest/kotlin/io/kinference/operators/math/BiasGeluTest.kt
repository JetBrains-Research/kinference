package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class BiasGeluTest {
    private fun getTargetPath(dirName: String) = "/biasgelu/$dirName/"

    @Test
    fun `test bias GELU with 1d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_1d_bias_gelu"))
    }

    @Test
    fun `test bias GELU with 2d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_2d_bias_gelu"))
    }

    @Test
    fun `test bias GELU with 3d data`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_3d_bias_gelu"))
    }
}
