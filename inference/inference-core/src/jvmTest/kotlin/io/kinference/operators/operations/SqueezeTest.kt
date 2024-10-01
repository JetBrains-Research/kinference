package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SqueezeVer1Test {
    private fun getTargetPath(dirName: String) = "squeeze/v1/$dirName/"

    @Test
    fun test_squeeze() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_squeeze"))
    }

    @Test
    fun test_squeeze_with_negative_axes() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_squeeze_negative_axes"))
    }
}

class SqueezeVer13Test {
    private fun getTargetPath(dirName: String) = "squeeze/v13/$dirName/"

    @Test
    fun test_squeeze() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_squeeze"))
    }

    @Test
    fun test_squeeze_with_negative_axes() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_squeeze_negative_axes"))
    }
}
