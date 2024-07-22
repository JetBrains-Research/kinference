package io.kinference.operators.layer.attention

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class QAttentionTest {
    private fun getTargetPath(dirName: String) = "qattention/$dirName/"

    @Test
    fun test_quantized_attention_defaults() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_qattention_defaults"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun test_quantized_attention() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_qattention_op"), delta = AccuracyRunner.QUANT_DELTA)
    }
}
