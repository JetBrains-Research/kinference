package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class GeluTest {
    private fun getTargetPath(dirName: String) = "/gelu/$dirName/"

    @Test
    fun `test GELU`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_gelu"))
    }
}
