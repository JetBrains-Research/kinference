package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class DivTest {
    private fun getTargetPath(dirName: String) = "/div/$dirName/"

    @Test
    fun test_div() = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_div"))
    }

    @Test
    fun test_div_broadcast() = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_div_bcast"))
    }

    @Test
    fun test_div_example() = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_div_example"))
    }

    @Test
    fun test_div_uint8() = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_div_uint8"))
    }
}
