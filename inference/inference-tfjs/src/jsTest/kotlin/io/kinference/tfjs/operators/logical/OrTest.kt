package io.kinference.tfjs.operators.logical

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class OrTest {
    private fun getTargetPath(dirName: String) = "or/$dirName/"

    @Test
    fun test_or_2d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_or2d"))
    }

    @Test
    fun test_or_3d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_or3d"))
    }

    @Test
    fun test_or_4d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_or4d"))
    }

    @Test
    fun test_or_broadcast_3v1d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_or_bcast3v1d"))
    }

    @Test
    fun test_or_broadcast_3v2d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_or_bcast3v2d"))
    }

    @Test
    fun test_or_broadcast_4v2d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_or_bcast4v2d"))
    }

    @Test
    fun test_or_broadcast_4v3d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_or_bcast4v3d"))
    }

    @Test
    fun test_or_broadcast_4v4d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_or_bcast4v4d"))
    }
}
