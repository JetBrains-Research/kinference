package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ScatterElementsTest {
    private fun getTargetPath(dirName: String) = "scatter_elements/$dirName/"

    @Test
    fun test_scatter_elements_with_axis() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_scatter_elements_with_axis"))
    }

    @Test
    fun test_scatter_elements_without_axis() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_scatter_elements_without_axis"))
    }

    @Test
    fun test_scatter_elements_with_negative_indices() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_scatter_elements_with_negative_indices"))
    }
}
