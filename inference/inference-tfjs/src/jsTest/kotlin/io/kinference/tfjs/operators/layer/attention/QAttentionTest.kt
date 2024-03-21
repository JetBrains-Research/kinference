package io.kinference.tfjs.operators.layer.attention

import io.kinference.runners.AccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class QAttentionTest {
    private fun getTargetPath(dirName: String) = "qattention/$dirName/"

    @Test
    fun test_quantized_attention_defaults()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_qattention_defaults"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun test_quantized_attention()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_qattention_op"), delta = AccuracyRunner.QUANT_DELTA)
    }
}
