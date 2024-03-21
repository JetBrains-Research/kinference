package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MaxTest {
    private fun getTargetPath(dirName: String) = "max/$dirName/"

    @Test
    fun test_max_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_example"))
    }

    @Test
    fun test_max_float16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_float16"))
    }

    @Test
    fun test_max_float32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_float32"))
    }

    @Test
    fun test_max_float64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_float64"))
    }

    @Test
    fun test_max_int8() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_int8"))
    }

    @Test
    fun test_max_int16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_int16"))
    }

    @Test
    fun test_max_int32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_int32"))
    }

    @Test
    fun test_max_int64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_int64"))
    }

    @Test
    fun test_max_one_input() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_one_input"))
    }

    @Test
    fun test_max_two_inputs() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_two_inputs"))
    }

    @Test
    fun test_max_uint8() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_uint8"))
    }

    @Test
    fun test_max_uint16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_uint16"))
    }

    @Test
    fun test_max_uint32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_uint32"))
    }

    @Test
    fun test_max_uint64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_max_uint64"))
    }
}
