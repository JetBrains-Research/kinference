package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SqueezeTest {
    private fun getTargetPath(dirName: String) = "squeeze/$dirName/"

    @Test
    fun test_squeeze() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_squeeze"))
    }

    @Test
    fun test_squeeze_with_negative_axes() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_squeeze_negative_axes"))
    }
}
