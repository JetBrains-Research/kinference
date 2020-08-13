package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class LSTMLayerTest {
    private fun getTargetPath(dirName: String) = "/lstm/$dirName/"

    @Test
    fun `test LSTM defaults`() {
        Utils.tensorTestRunner(getTargetPath("test_lstm_defaults"))
    }

    @Test
    fun `test LSTM with initial bias`() {
        Utils.tensorTestRunner(getTargetPath("test_lstm_with_initial_bias"))
    }

    @Test
    fun `test BiLSTM defaults`() {
        Utils.tensorTestRunner(getTargetPath("test_bilstm_defaults"))
    }

    @Test
    fun `test BiLSTM with bias`() {
        Utils.tensorTestRunner(getTargetPath("test_bilstm_with_bias"))
    }
}
