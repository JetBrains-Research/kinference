package org.jetbrains.research.kotlin.inference.operators.layer.attention

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class AttentionTest {
    private fun getTargetPath(dirName: String) = "/attention/$dirName/"

    @Test
    fun `test unidirectional multi-head masked attention`() {
        Utils.tensorTestRunner(getTargetPath("test_unidirectional_masked_multi_head"))
    }
}
