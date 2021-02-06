package io.kinference.operators.math

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class MulTest {
    private fun getTargetPath(dirName: String) = "/mul/$dirName/"

    @Test
    fun `test mul`() {
        TestRunner.runFromResources(getTargetPath("test_mul"))
    }

    @Test
    fun `test mul with broadcast`() {
        TestRunner.runFromResources(getTargetPath("test_mul_bcast"))
    }

    @Test
    fun `test mul defaults`() {
        TestRunner.runFromResources(getTargetPath("test_mul_example"))
    }
}
