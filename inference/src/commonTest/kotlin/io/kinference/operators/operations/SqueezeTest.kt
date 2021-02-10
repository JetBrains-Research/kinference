package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SqueezeTest {
    private fun getTargetPath(dirName: String) = "/squeeze/$dirName/"

    @Test
    fun test_squeeze()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_squeeze"))
    }

    @Test
    fun test_squeeze_with_negative_axes()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_squeeze_negative_axes"))
    }
}
