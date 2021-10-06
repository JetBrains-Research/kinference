package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "/add/$dirName/"

    @Test
    fun test_add()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_add"))
    }

    @Test
    fun test_add_broadcast()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_add_bcast"))
    }

    @Test
    fun test_add_scalar()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_add_scalar"))
    }
}
