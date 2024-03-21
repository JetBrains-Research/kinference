package io.kinference.operators.layer.normalization

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class BatchNormalizationTest {
    private fun getTargetPath(dirName: String) = "batch_normalization/$dirName/"

    @Test
    fun test_batch_normalization_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_batchnorm_example"))
    }

    @Test
    fun test_batch_normalization_epsilon() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_batchnorm_epsilon"))
    }
}
