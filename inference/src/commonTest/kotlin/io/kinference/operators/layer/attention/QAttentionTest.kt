package io.kinference.operators.layer.attention

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.math.pow
import kotlin.test.Test

class QAttentionTest {
    private fun getTargetPath(dirName: String) = "/qattention/$dirName/"

    @Test
    fun test_quantized_attention_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_qattention_defaults"), delta = AccuracyRunner.quantDelta)
    }

    @Test
    fun test_quantized_attention()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_qattention_op"), delta = AccuracyRunner.quantDelta)
    }
}
