package io.kinference.operators.layer.normalization

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class EmbedLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "embed_layer_normalization/$dirName/"

    @Test
    fun test_embedding_layer_normalization_defaults() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_all_inputs"))
    }

    @Test
    fun test_unmasked_embedding_layer_normalization() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_no_mask"))
    }

    @Test
    fun test_embedding_layer_normalization_with_epsilon() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_all_inputs_with_epsilon"))
    }
}
