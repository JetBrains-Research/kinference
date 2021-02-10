package io.kinference.operators.flow

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class LoopTest {
    private fun getTargetPath(dirName: String) = "/loop/$dirName/"

    @Test
    fun `test loop`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_loop"))
    }
}
