package io.kinference.operators.layer.normalization

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class EmbedLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/embed_layer_normalization/$dirName/"

    @Test
    fun `test embedding layer normalization defaults`() {
        TestRunner.runFromResources(getTargetPath("test_all_inputs"))
    }

    @Test
    fun `test unmasked embedding layer normalization`() {
        TestRunner.runFromResources(getTargetPath("test_no_mask"))
    }

    @Test
    fun `test embedding layer normalization with epsilon`() {
        TestRunner.runFromResources(getTargetPath("test_all_inputs_with_epsilon"))
    }
}
