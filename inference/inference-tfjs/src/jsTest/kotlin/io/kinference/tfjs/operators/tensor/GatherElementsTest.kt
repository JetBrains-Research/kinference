package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class GatherElementsTest {
    private fun getTargetPath(dirName: String) = "gather_elements/$dirName/"

    @Test
    fun test_gather_elements_0() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_0"))
    }

    @Test
    fun test_gather_elements_1() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_1"))
    }

    @Test
    fun test_gather_elements_with_negative_indices() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_negative_indices"))
    }

    @Test
    fun test_gather_elements_model() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_model"))
    }
}
