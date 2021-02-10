package io.kinference.operators.flow

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class WhereTest {
    private fun getTargetPath(dirName: String) = "/where/$dirName/"

    @Test
    fun `test where`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_where_example"))
    }
}
