package io.kinference.tfjs.operators.layer.recurrent.gru

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class GRULayerTest {
    private fun getTargetPath(dirName: String) = "gru/$dirName/"

    @Test
    fun test_GRU_defaults() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gru_defaults"))
    }

    @Test
    fun test_GRU_with_initial_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gru_with_initial_bias"))
    }

    @Test
    fun test_GRU_seq_length() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gru_seq_length"))
    }

    @Test
    fun test_GRU_bidirectional() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gru_bidirectional"))
    }
}
