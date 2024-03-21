package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class PadTest {
    private fun getTargetPath(dirName: String) = "pad/$dirName/"

    @Test
    fun test_constant_pad()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_constant_pad"))
    }

    @Test
    fun test_edge_pad()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_edge_pad"))
    }

    @Test
    fun test_reflect_pad()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reflect_pad"))
    }
}
