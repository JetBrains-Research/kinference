package io.kinference.operators.layer.recurrent.lstm

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class LSTMLayerTest {
    private fun getTargetPath(dirName: String) = "/lstm/$dirName/"

    @Test
    fun `test LSTM defaults`() {
        TestRunner.runFromResources(getTargetPath("test_lstm_defaults"))
    }

    @Test
    fun `test LSTM with initial bias`() {
        TestRunner.runFromResources(getTargetPath("test_lstm_with_initial_bias"))
    }

    @Test
    fun `test BiLSTM defaults`() {
        TestRunner.runFromResources(getTargetPath("test_bilstm_defaults"))
    }

    @Test
    fun `test BiLSTM with bias`() {
        TestRunner.runFromResources(getTargetPath("test_bilstm_with_bias"))
    }
}
