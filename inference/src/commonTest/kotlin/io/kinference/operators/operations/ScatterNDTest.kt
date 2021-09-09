package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class ScatterNDTest {
    private fun getTargetPath(dirName: String) = "/scatter_nd/$dirName/"

    @Test
    fun test_scatter_nd() = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_scatternd"))
    }
}
