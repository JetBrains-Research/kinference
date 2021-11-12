package io.kinference.operators.operations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GatherElementsTest {
    private fun getTargetPath(dirName: String) = "/gather_elements/$dirName/"

    @Test
    fun test_gather_elements_0() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_0"))
    }

    @Test
    fun test_gather_elements_1() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_1"))
    }

    @Test
    fun test_gather_elements_with_negative_indices() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_gather_elements_negative_indices"))
    }
}
