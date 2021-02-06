package io.kinference.operators.layer.attention

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class AttentionTest {
    private fun getTargetPath(dirName: String) = "/attention/$dirName/"

    @Test
    fun `test unidirectional multi-head masked attention`() {
        TestRunner.runFromResources(getTargetPath("test_unidirectional_masked_multi_head"))
    }
}
