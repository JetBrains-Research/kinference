package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ScatterNDTest {
    private fun getTargetPath(dirName: String) = "scatter_nd/$dirName/"

    @Test
    fun test_scatter_nd() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_scatternd"))
    }
}
