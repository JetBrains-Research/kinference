package io.kinference.operators.flow

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class WhereTest {
    private fun getTargetPath(dirName: String) = "where/$dirName/"

    @Test
    fun test_where() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_where_example"))
    }
}
