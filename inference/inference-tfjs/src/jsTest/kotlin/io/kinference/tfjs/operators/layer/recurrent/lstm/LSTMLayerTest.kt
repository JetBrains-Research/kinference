package io.kinference.tfjs.operators.layer.recurrent.lstm

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LSTMLayerTest {
    private fun getTargetPath(dirName: String) = "lstm/$dirName/"

    @Test
    fun test_LSTM_defaults() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_lstm_defaults"))
    }

    @Test
    fun test_LSTM_with_initial_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_lstm_with_initial_bias"))
    }

    @Test
    fun test_LSTM_with_peepholes() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_lstm_with_peepholes"))
    }

    @Test
    fun test_BiLSTM_defaults() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_bilstm_defaults"))
    }

    @Test
    fun test_BiLSTM_with_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_bilstm_with_bias"))
    }
}
