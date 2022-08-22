package io.kinference.tfjs.operators.flow

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class WhereTest {
    private fun getTargetPath(dirName: String) = "where/$dirName/"

    @Test
    fun test_where() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_where_example"))
    }
}
