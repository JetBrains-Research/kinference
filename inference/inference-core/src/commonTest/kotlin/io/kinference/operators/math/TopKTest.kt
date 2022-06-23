package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TopKTest {
    private fun getTargetPath(dirName: String) = "top_k/$dirName/"

    @Test
    fun test_top_k()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_top_k"))
    }

    @Test
    fun test_top_k_negative_axis()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_top_k_negative_axis"))
    }

    @Test
    fun test_top_k_smallest()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_top_k_smallest"))
    }
}
