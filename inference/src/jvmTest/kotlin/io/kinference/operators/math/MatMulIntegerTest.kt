package io.kinference.operators.math

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class MatMulIntegerTest {
    private fun getTargetPath(dirName: String) = "/matmul_integer/$dirName/"

    @Test
    fun `test matmul integer`() {
        TestRunner.runFromResources(getTargetPath("test_matmulinteger"))
    }

    @Test
    fun `test matmul integer op`() {
        TestRunner.runFromResources(getTargetPath("test_matmulinteger_op"))
    }
}
