package io.kinference.operators.operations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MinTest {
    private fun getTargetPath(dirName: String) = "min/$dirName/"

    @Test
    fun test_min_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_example"))
    }

    @Test
    fun test_min_float16() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_float16"))
    }

    @Test
    fun test_min_float32() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_float32"))
    }

    @Test
    fun test_min_float64() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_float64"))
    }

    @Test
    fun test_min_int8() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_int8"))
    }


    @Test
    fun test_min_int16() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_int16"))
    }

    @Test
    fun test_min_int32() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_int32"))
    }

    @Test
    fun test_min_int64() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_int64"))
    }

    @Test
    fun test_min_one_input() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_one_input"))
    }

    @Test
    fun test_min_two_inputs() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_two_inputs"))
    }

    @Test
    fun test_min_uint8() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_uint8"))
    }

    @Test
    fun test_min_uint16() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_uint16"))
    }

    @Test
    fun test_min_uint32() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_uint32"))
    }

    @Test
    fun test_min_uint64() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_min_uint64"))
    }
}
