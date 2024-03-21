package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SubTest {
    private fun getTargetPath(dirName: String) = "sub/$dirName/"

    @Test
    fun test_sub()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sub"))
    }

    @Test
    fun test_sub_broadcast()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sub_bcast"))
    }

    @Test
    fun test_sub_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sub_example"))
    }

    @Test
    fun test_sub_uint8()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sub_uint8"))
    }
}
