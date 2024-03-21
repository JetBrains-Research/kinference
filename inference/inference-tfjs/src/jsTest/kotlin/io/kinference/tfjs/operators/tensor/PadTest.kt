package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class PadTest {
    private fun getTargetPath(dirName: String) = "pad/$dirName/"

    @Test
    fun test_constant_pad()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_constant_pad"))
    }

    @Test
    fun test_edge_pad()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_edge_pad"))
    }

    @Test
    fun test_reflect_pad()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reflect_pad"))
    }
}
