package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class TopKTest {
    private fun getTargetPath(dirName: String) = "top_k/$dirName/"

    @Test
    fun test_top_k()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_top_k"))
    }

    @Test
    fun test_top_k_negative_axis()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_top_k_negative_axis"))
    }

    @Test
    fun test_top_k_smallest()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_top_k_smallest"))
    }
}
