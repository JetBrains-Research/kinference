package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class DivTest {
    private fun getTargetPath(dirName: String) = "div/$dirName/"

    @Test
    fun test_div() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_div"))
    }

    @Test
    fun test_div_broadcast() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_div_bcast"))
    }

    @Test
    fun test_div_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_div_example"))
    }

    @Test
    fun test_div_uint8() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_div_uint8"))
    }
}
