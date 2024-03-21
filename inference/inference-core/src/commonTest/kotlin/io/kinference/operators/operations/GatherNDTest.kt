package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class GatherNDTest {
    private fun getTargetPath(dirName: String) = "gather_nd/$dirName/"

    @Test
    fun test_gather_nd_float32()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gathernd_example_float32"))
    }

    @Test
    fun test_gather_nd_int32()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gathernd_example_int32"))
    }

    @Test
    fun test_gather_nd_int32_batch_dim_1()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gathernd_example_int32_batch_dim1"))
    }

    @Test
    fun test_gather_nd_batch_dim_2() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gathernd_batch_dim2"))
    }
}
