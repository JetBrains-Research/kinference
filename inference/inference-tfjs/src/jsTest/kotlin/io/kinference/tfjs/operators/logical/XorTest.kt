package io.kinference.tfjs.operators.logical

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class XorTest {
    private fun getTargetPath(dirName: String) = "xor/$dirName/"

    @Test
    fun test_xor_2d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_xor2d"))
    }

    @Test
    fun test_xor_3d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_xor3d"))
    }

    @Test
    fun test_xor_4d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_xor4d"))
    }

    @Test
    fun test_xor_broadcast_3v1d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_xor_bcast3v1d"))
    }

    @Test
    fun test_xor_broadcast_3v2d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_xor_bcast3v2d"))
    }

    @Test
    fun test_xor_broadcast_4v2d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_xor_bcast4v2d"))
    }

    @Test
    fun test_xor_broadcast_4v3d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_xor_bcast4v3d"))
    }

    @Test
    fun test_xor_broadcast_4v4d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_xor_bcast4v4d"))
    }
}
