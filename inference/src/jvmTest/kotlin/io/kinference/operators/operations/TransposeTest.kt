package io.kinference.operators.operations

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class TransposeTest {
    private fun getTargetPath(dirName: String) = "/transpose/$dirName/"

    @Test
    fun `test transpose all permutations 0`() {
        TestRunner.runFromResources(getTargetPath("test_transpose_all_permutations_0"))
    }

    @Test
    fun `test transpose all permutations 1`() {
        TestRunner.runFromResources(getTargetPath("test_transpose_all_permutations_1"))
    }

    @Test
    fun `test transpose all permutations 2`() {
        TestRunner.runFromResources(getTargetPath("test_transpose_all_permutations_2"))
    }

    @Test
    fun `test transpose all permutations 3`() {
        TestRunner.runFromResources(getTargetPath("test_transpose_all_permutations_3"))
    }

    @Test
    fun `test transpose all permutations 4`() {
        TestRunner.runFromResources(getTargetPath("test_transpose_all_permutations_4"))
    }

    @Test
    fun `test transpose all permutations 5`() {
        TestRunner.runFromResources(getTargetPath("test_transpose_all_permutations_5"))
    }

    @Test
    fun `test transpose default`() {
        TestRunner.runFromResources(getTargetPath("test_transpose_default"))
    }
}
