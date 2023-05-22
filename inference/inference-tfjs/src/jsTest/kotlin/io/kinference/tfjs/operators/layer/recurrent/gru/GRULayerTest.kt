package io.kinference.tfjs.operators.layer.recurrent.gru

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GRULayerTest {
    private fun getTargetPath(dirName: String) = "gru/$dirName/"

    @Test
    fun test_GRU_defaults() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gru_defaults"))
    }

    @Test
    fun test_GRU_with_initial_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gru_with_initial_bias"))
    }

    @Test
    fun test_GRU_seq_length() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gru_seq_length"))
    }
}
