package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class MulTest {
    private fun getTargetPath(dirName: String) = "/mul/$dirName/"

    @Test
    fun `test mul`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_mul"))
    }

    @Test
    fun `test mul with broadcast`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_mul_bcast"))
    }

    @Test
    fun `test mul defaults`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_mul_example"))
    }
}
