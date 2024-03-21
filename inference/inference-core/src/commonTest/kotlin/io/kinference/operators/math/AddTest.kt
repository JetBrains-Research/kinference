package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "add/$dirName/"

    @Test
    fun test_add() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_add"))
    }

    @Test
    fun test_add_broadcast() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_add_bcast"))
    }

    @Test
    fun test_add_scalar() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_add_scalar"))
    }
}
