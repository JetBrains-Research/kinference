package io.kinference.tfjs.operators.logical

import io.kinference.tfjs.runners.TFJSTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AndTest {
    private fun getTargetPath(dirName: String) = "and/$dirName/"

    @Test
    fun test_and_2d() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_and2d"))
    }

    @Test
    fun test_and_3d() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_and3d"))
    }

    @Test
    fun test_and_4d() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_and4d"))
    }

    @Test
    fun test_and_broadcast_3v1d() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_and_bcast3v1d"))
    }

    @Test
    fun test_and_broadcast_3v2d() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_and_bcast3v2d"))
    }

    @Test
    fun test_and_broadcast_4v2d() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_and_bcast4v2d"))
    }

    @Test
    fun test_and_broadcast_4v3d() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_and_bcast4v3d"))
    }

    @Test
    fun test_and_broadcast_4v4d() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_and_bcast4v4d"))
    }
}
