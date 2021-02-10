package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class ConstantTest {
    private fun getTargetPath(dirName: String) = "/constant/$dirName/"

    @Test
    fun `test constant`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_constant"))
    }

    @Test
    fun `test scalar constant`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_scalar_constant"))
    }
}
