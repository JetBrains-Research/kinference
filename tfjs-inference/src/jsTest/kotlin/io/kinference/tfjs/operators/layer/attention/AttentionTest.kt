package io.kinference.tfjs.operators.layer.attention

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class AttentionTest {
    private fun getTargetPath(dirName: String) = "/attention/$dirName/"

    @Test
    fun test_unidirectional_multi_head_masked_attention()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_unidirectional_masked_multi_head"))
    }
}
