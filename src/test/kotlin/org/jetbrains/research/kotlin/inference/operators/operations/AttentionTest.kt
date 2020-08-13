package org.jetbrains.research.kotlin.inference.operators.operations

import org.jetbrains.research.kotlin.inference.Utils
import org.jetbrains.research.kotlin.inference.extensions.ndarray.transpose
import org.jetbrains.research.kotlin.inference.extensions.primitives.gemm
import org.junit.jupiter.api.Test

class AttentionTest {
    private fun getTargetPath(dirName: String) = "/attention/$dirName/"

    @Test
    fun `test unidirectional multi-head masked attention`() {
        Utils.tensorTestRunner(getTargetPath("test_unidirectional_masked_multi_head"))
    }
}
