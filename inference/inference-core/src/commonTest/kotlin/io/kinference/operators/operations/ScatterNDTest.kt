package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ScatterNDTest {
    private fun getTargetPath(dirName: String) = "scatter_nd/$dirName/"

    @Test
    fun test_scatter_nd() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_scatternd"))
    }
}
