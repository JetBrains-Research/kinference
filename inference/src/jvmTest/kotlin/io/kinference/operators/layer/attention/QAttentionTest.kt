package io.kinference.operators.layer.attention

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test
import kotlin.math.pow

class QAttentionTest {
    private fun getTargetPath(dirName: String) = "/qattention/$dirName/"

    @Test
    fun `test quantized attention defaults`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_qattention_defaults"), delta = (10.0).pow(-2))
    }

    @Test
    fun `test quantized attention`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_qattention_op"))
    }
}
