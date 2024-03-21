package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MaxTest {
    private fun getTargetPath(dirName: String) = "max/$dirName/"

    @Test
    fun test_max_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_example"))
    }

    @Test
    fun test_max_float16() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_float16"))
    }

    @Test
    fun test_max_float32() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_float32"))
    }

    @Test
    fun test_max_float64() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_float64"))
    }

    @Test
    fun test_max_int8() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_int8"))
    }


    @Test
    fun test_max_int16() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_int16"))
    }

    @Test
    fun test_max_int32() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_int32"))
    }

    @Test
    fun test_max_int64() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_int64"))
    }

    @Test
    fun test_max_one_input() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_one_input"))
    }

    @Test
    fun test_max_two_inputs() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_two_inputs"))
    }

    @Test
    fun test_max_uint8() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_uint8"))
    }

    @Test
    fun test_max_uint16() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_uint16"))
    }

    @Test
    fun test_max_uint32() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_uint32"))
    }

    @Test
    fun test_max_uint64() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_uint64"))
    }
}
