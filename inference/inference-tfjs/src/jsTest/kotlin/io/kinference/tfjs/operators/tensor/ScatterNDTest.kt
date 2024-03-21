package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ScatterNDTest {
    private fun getTargetPath(dirName: String) = "scatter_nd/$dirName/"

    @Test
    fun test_scatter_nd() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_scatternd"))
    }
}
