package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class BiasGeluTest {
    private fun getTargetPath(dirName: String) = "/biasgelu/$dirName/"

    @Test
    fun test_bias_GELU_with_1d_data()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_1d_bias_gelu"))
    }

    @Test
    fun test_bias_GELU_with_2d_data()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_2d_bias_gelu"))
    }

    @Test
    fun test_bias_GELU_with_3d_data()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_3d_bias_gelu"))
    }
}
