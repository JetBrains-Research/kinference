package io.kinference.tfjs.operators.tensor

import io.kinference.utils.TestRunner
import kotlin.test.Test
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner


class GatherElementsTest {
    private fun getTargetPath(dirName: String) = "gather_elements/$dirName/"

    @Test
    fun test_gather_elements_0() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_0"))
    }

    @Test
    fun test_gather_elements_1() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_1"))
    }

    @Test
    fun test_gather_elements_with_negative_indices() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_negative_indices"))
    }

    @Test
    fun test_gather_elements_model() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_model"))
    }
}
