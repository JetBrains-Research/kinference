package io.kinference.tfjs.operators.layer.normalization

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class BatchNormalizationTest {
    private fun getTargetPath(dirName: String) = "batch_normalization/$dirName/"

    @Test
    fun test_batch_normalization_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_batchnorm_example"))
    }

    @Test
    fun test_batch_normalization_epsilon() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_batchnorm_epsilon"))
    }
}
