package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SubTest {
    private fun getTargetPath(dirName: String) = "sub/$dirName/"

    @Test
    fun test_sub()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sub"))
    }

    @Test
    fun test_sub_broadcast()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sub_bcast"))
    }

    @Test
    fun test_sub_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sub_example"))
    }

    @Test
    fun test_sub_uint8()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sub_uint8"))
    }
}
