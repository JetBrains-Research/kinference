package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class DetTest {
    private fun getTargetPath(dirName: String) = "det/$dirName/"

    @Test
    fun test_det_2d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_det_2d"))
    }

    @Test
    fun test_det_nd() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_det_nd"))
    }
}
