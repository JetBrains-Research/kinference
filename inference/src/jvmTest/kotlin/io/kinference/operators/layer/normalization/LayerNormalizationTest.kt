package io.kinference.operators.layer.normalization

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class LayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/layer_normalization/$dirName/"

    @Test
    fun `test layer normalization defaults`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_layer_normalization_0"))
    }

    @Test
    fun `test with negative axis`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_negate_axis"))
    }
}
