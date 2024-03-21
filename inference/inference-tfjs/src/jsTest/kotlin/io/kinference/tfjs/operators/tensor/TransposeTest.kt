package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class TransposeTest {
    private fun getTargetPath(dirName: String) = "transpose/$dirName/"

    @Test
    fun test_transpose_all_permutations_0() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_0"))
    }

    @Test
    fun test_transpose_all_permutations_1() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_1"))
    }

    @Test
    fun test_transpose_all_permutations_2() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_2"))
    }

    @Test
    fun test_transpose_all_permutations_3() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_3"))
    }

    @Test
    fun test_transpose_all_permutations_4() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_4"))
    }

    @Test
    fun test_transpose_all_permutations_5() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_5"))
    }

    @Test
    fun test_transpose_default() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_transpose_default"))
    }
}
