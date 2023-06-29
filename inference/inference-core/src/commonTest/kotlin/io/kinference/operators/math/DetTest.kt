package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class DetTest {
    private fun getTargetPath(dirName: String) = "det/$dirName/"

    @Test
    fun test_det_2d() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_det_2d"))
    }

    @Test
    fun test_det_nd() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_det_nd"))
    }
}
