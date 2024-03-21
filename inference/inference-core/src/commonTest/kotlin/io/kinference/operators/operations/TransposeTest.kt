package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TransposeTest {
    private fun getTargetPath(dirName: String) = "transpose/$dirName/"

    @Test
    fun test_transpose_all_permutations_0() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_0"))
    }

    @Test
    fun test_transpose_all_permutations_1() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_1"))
    }

    @Test
    fun test_transpose_all_permutations_2() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_2"))
    }

    @Test
    fun test_transpose_all_permutations_3() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_3"))
    }

    @Test
    fun test_transpose_all_permutations_4() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_4"))
    }

    @Test
    fun test_transpose_all_permutations_5() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_5"))
    }

    @Test
    fun test_transpose_default() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_default"))
    }
}
