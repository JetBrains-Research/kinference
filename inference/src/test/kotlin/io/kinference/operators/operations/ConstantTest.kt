package io.kinference.operators.operations

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class ConstantTest {
    private fun getTargetPath(dirName: String) = "/constant/$dirName/"

    @Test
    fun `test constant`() {
        TestRunner.runFromResources(getTargetPath("test_constant"))
    }

    @Test
    fun `test scalar constant`() {
        TestRunner.runFromResources(getTargetPath("test_scalar_constant"))
    }
}
