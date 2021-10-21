package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TransposeTest {
    private fun getTargetPath(dirName: String) = "/transpose/$dirName/"

    @Test
    fun test_transpose_all_permutations_0() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_0"))
    }

    @Test
    fun test_transpose_all_permutations_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_1"))
    }

    @Test
    fun test_transpose_all_permutations_2() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_2"))
    }

    @Test
    fun test_transpose_all_permutations_3() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_3"))
    }

    @Test
    fun test_transpose_all_permutations_4() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_4"))
    }

    @Test
    fun test_transpose_all_permutations_5() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_all_permutations_5"))
    }

    @Test
    fun test_transpose_default() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_transpose_default"))
    }
}
