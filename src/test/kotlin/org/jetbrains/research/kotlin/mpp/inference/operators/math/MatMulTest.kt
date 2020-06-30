package org.jetbrains.research.kotlin.mpp.inference.operators.math

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test
import java.io.File

class MatMulTest {
    private fun getTargetPath(dirName: String) = "/matmul/$dirName/"

    @Test
    fun test_matmul_2d() {
        Utils.singleTestHelper(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun test_matmul_3d(){
        Utils.singleTestHelper(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun test_matmul_4d(){
        Utils.singleTestHelper(getTargetPath("test_matmul_4d"))
    }
}
