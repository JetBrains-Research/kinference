package org.jetbrains.research.kotlin.inference.operators.operations

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class TransposeTest {
    private fun getTargetPath(dirName: String) = "/transpose/$dirName/"

    @Test
    fun test_transpose_all_permutations_0() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_0"))
    }

    @Test
    fun test_transpose_all_permutations_1() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_1"))
    }

    @Test
    fun test_transpose_all_permutations_2() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_2"))
    }

    @Test
    fun test_transpose_all_permutations_3() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_3"))
    }

    @Test
    fun test_transpose_all_permutations_4() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_4"))
    }

    @Test
    fun test_transpose_all_permutations_5() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_all_permutations_5"))
    }

    @Test
    fun test_transpose_default() {
        Utils.tensorTestRunner(getTargetPath("test_transpose_default"))
    }
}
