package io.kinference.operators.flow

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class WhereTest {
    private fun getTargetPath(dirName: String) = "/where/$dirName/"

    @Test
    fun test_where()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_where_example"))
    }
}
