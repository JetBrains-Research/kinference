package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MinTest {
    private fun getTargetPath(dirName: String) = "min/$dirName/"

    @Test
    fun test_min_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_example"))
    }

    @Test
    fun test_min_float16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_float16"))
    }

    @Test
    fun test_min_float32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_float32"))
    }

    @Test
    fun test_min_float64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_float64"))
    }

    @Test
    fun test_min_int8() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_int8"))
    }


    @Test
    fun test_min_int16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_int16"))
    }

    @Test
    fun test_min_int32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_int32"))
    }

    @Test
    fun test_min_int64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_int64"))
    }

    @Test
    fun test_min_one_input() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_one_input"))
    }

    @Test
    fun test_min_two_inputs() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_two_inputs"))
    }

    @Test
    fun test_min_uint8() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_uint8"))
    }

    @Test
    fun test_min_uint16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_uint16"))
    }

    @Test
    fun test_min_uint32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_uint32"))
    }

    @Test
    fun test_min_uint64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_min_uint64"))
    }
}
