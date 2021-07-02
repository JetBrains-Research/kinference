package io.kinference.operators.layer.attention

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

class QAttentionTest {
    private fun getTargetPath(dirName: String) = "/qattention/$dirName/"

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_quantized_attention_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_qattention_defaults"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_quantized_attention()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_qattention_op"), delta = AccuracyRunner.QUANT_DELTA)
    }
}
