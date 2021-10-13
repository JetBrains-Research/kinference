package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "/add/$dirName/"

    @Test
    fun test_add()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_add"))
    }

    @Test
    fun test_add_broadcast()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_add_bcast"))
    }

    @Test
    fun test_add_scalar()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_add_scalar"))
    }
}
