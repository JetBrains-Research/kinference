package io.kinference.tfjs.operators.layer.attention

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class QAttentionTest {
    private fun getTargetPath(dirName: String) = "/qattention/$dirName/"

    @Test
    fun test_quantized_attention_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_qattention_defaults"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun test_quantized_attention()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_qattention_op"), delta = AccuracyRunner.QUANT_DELTA)
    }
}
