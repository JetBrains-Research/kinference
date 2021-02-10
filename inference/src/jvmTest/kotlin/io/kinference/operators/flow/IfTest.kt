package io.kinference.operators.flow

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class IfTest {
    private fun getTargetPath(dirName: String) = "/if/$dirName/"

    @Test
    fun `test if`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_if"))
    }
}
