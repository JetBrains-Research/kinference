package io.kinference.tfjs.operators.flow

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class WhereTest {
    private fun getTargetPath(dirName: String) = "where/$dirName/"

    @Test
    fun test_where() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_where_example"))
    }
}
