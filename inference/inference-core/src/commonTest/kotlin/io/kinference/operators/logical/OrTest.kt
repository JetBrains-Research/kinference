package io.kinference.operators.logical

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class OrTest {
    private fun getTargetPath(dirName: String) = "or/$dirName/"

    @Test
    fun test_or_2d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_or2d"))
    }

    @Test
    fun test_or_3d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_or3d"))
    }

    @Test
    fun test_or_4d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_or4d"))
    }

    @Test
    fun test_or_broadcast_3v1d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_or_bcast3v1d"))
    }

    @Test
    fun test_or_broadcast_3v2d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_or_bcast3v2d"))
    }

    @Test
    fun test_or_broadcast_4v2d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_or_bcast4v2d"))
    }

    @Test
    fun test_or_broadcast_4v3d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_or_bcast4v3d"))
    }

    @Test
    fun test_or_broadcast_4v4d() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_or_bcast4v4d"))
    }
}
