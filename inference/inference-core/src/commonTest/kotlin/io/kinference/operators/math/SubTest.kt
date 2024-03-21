package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SubTest {
    private fun getTargetPath(dirName: String) = "sub/$dirName/"

    @Test
    fun test_sub()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sub"))
    }

    @Test
    fun test_sub_broadcast()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sub_bcast"))
    }

    @Test
    fun test_sub_example()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sub_example"))
    }

    @Test
    fun test_sub_uint8()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sub_uint8"))
    }
}
