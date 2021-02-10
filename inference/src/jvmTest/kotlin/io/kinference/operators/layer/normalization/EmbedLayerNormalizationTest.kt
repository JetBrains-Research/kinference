package io.kinference.operators.layer.normalization

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class EmbedLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/embed_layer_normalization/$dirName/"

    @Test
    fun `test embedding layer normalization defaults`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_all_inputs"))
    }

    @Test
    fun `test unmasked embedding layer normalization`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_no_mask"))
    }

    @Test
    fun `test embedding layer normalization with epsilon`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_all_inputs_with_epsilon"))
    }
}
