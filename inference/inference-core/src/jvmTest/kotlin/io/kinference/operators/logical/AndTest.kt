package io.kinference.operators.logical

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AndTest {
    private fun getTargetPath(dirName: String) = "and/$dirName/"

    @Test
    fun test_and_2d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_and2d"))
    }

    @Test
    fun test_and_3d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_and3d"))
    }

    @Test
    fun test_and_4d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_and4d"))
    }

    @Test
    fun test_and_broadcast_3v1d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_and_bcast3v1d"))
    }

    @Test
    fun test_and_broadcast_3v2d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_and_bcast3v2d"))
    }

    @Test
    fun test_and_broadcast_4v2d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_and_bcast4v2d"))
    }

    @Test
    fun test_and_broadcast_4v3d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_and_bcast4v3d"))
    }

    @Test
    fun test_and_broadcast_4v4d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_and_bcast4v4d"))
    }
}
