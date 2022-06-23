package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class PadTest {
    private fun getTargetPath(dirName: String) = "pad/$dirName/"

    @Test
    fun test_constant_pad()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_constant_pad"))
    }

    @Test
    fun test_edge_pad()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_edge_pad"))
    }

    @Test
    fun test_reflect_pad()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reflect_pad"))
    }
}
