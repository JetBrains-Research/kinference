package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ScatterElementsTest {
    private fun getTargetPath(dirName: String) = "scatter_elements/$dirName/"

    @Test
    fun test_scatter_elements_with_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_scatter_elements_with_axis"))
    }

    @Test
    fun test_scatter_elements_without_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_scatter_elements_without_axis"))
    }

    @Test
    fun test_scatter_elements_with_negative_indices() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_scatter_elements_with_negative_indices"))
    }
}
