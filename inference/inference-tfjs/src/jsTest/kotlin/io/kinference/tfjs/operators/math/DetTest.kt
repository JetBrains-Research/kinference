package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class DetTest {
    private fun getTargetPath(dirName: String) = "det/$dirName/"

    @Test
    fun test_det_2d() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_det_2d"))
    }

    @Test
    fun test_det_nd() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_det_nd"))
    }
}
