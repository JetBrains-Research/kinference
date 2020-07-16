package org.jetbrains.research.kotlin.mpp.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class LSTMLayerTest {
    private fun getTargetPath(dirName: String) = "/lstm/$dirName/"

    @Test
    fun test_lstm_defaults() {
        Utils.tensorTestRunner(getTargetPath("test_lstm_defaults"))
    }

    @Test
    fun test_lstm_with_initial_bias() {
        Utils.tensorTestRunner(getTargetPath("test_lstm_with_initial_bias"))
    }

    @Test
    fun test_bilstm_defaults() {
        Utils.tensorTestRunner(getTargetPath("test_bilstm_defaults"))
    }

    @Test
    fun test_bilstm_with_bias() {
        Utils.tensorTestRunner(getTargetPath("test_bilstm_with_bias"))
    }
}
