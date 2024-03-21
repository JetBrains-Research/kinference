package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "add/$dirName/"

    @Test
    fun test_add()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_add"))
    }

    @Test
    fun test_add_broadcast()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_add_bcast"))
    }

    @Test
    fun test_add_scalar()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_add_scalar"))
    }
}
