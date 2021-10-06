package io.kinference.tfjs.operators.layer.normalization

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class EmbedLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/embed_layer_normalization/$dirName/"

    @Test
    fun test_embedding_layer_normalization_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_all_inputs"))
    }

    @Test
    fun test_unmasked_embedding_layer_normalization()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_no_mask"))
    }

    @Test
    fun test_embedding_layer_normalization_with_epsilon()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_all_inputs_with_epsilon"))
    }
}
