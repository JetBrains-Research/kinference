package org.jetbrains.research.kotlin.inference.operators.operations

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class TransposeTest {
    private fun getTargetPath(dirName: String) = "/transpose/$dirName/"

    @Test
    fun `test transpose all permutations 0`() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_0"))
    }

    @Test
    fun `test transpose all permutations 1`() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_1"))
    }

    @Test
    fun `test transpose all permutations 2`() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_2"))
    }

    @Test
    fun `test transpose all permutations 3`() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_3"))
    }

    @Test
    fun `test transpose all permutations 4`() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_4"))
    }

    @Test
    fun `test transpose all permutations 5`() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_5"))
    }

    @Test
    fun `test transpose default`() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_default"))
    }
}
