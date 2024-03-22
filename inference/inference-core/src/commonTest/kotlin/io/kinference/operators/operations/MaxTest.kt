package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MaxTest {
    private fun getTargetPath(dirName: String) = "max/$dirName/"

    @Test
    fun test_max_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_example"))
    }

    @Test
    fun test_max_float16() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_float16"))
    }

    @Test
    fun test_max_float32() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_float32"))
    }

    @Test
    fun test_max_float64() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_float64"))
    }

    @Test
    fun test_max_int8() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_int8"))
    }


    @Test
    fun test_max_int16() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_int16"))
    }

    @Test
    fun test_max_int32() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_int32"))
    }

    @Test
    fun test_max_int64() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_int64"))
    }

    @Test
    fun test_max_one_input() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_one_input"))
    }

    @Test
    fun test_max_two_inputs() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_two_inputs"))
    }

    @Test
    fun test_max_uint8() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_uint8"))
    }

    @Test
    fun test_max_uint16() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_uint16"))
    }

    @Test
    fun test_max_uint32() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_uint32"))
    }

    @Test
    fun test_max_uint64() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_max_uint64"))
    }
}
