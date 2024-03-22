package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SubTest {
    private fun getTargetPath(dirName: String) = "sub/$dirName/"

    @Test
    fun test_sub()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sub"))
    }

    @Test
    fun test_sub_broadcast()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sub_bcast"))
    }

    @Test
    fun test_sub_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sub_example"))
    }

    @Test
    fun test_sub_uint8()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sub_uint8"))
    }
}
