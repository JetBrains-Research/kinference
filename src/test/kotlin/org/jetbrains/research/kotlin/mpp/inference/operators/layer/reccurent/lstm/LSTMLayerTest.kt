package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class LSTMLayerTest {
    private fun getTargetPath(dirName: String) = "/lstm/$dirName/"

    @Test
    fun test_lstm_defaults() {
        Utils.singleTestHelper(getTargetPath("test_lstm_defaults"))
    }

    @Test
    fun test_lstm_with_initial_bias() {
        Utils.singleTestHelper(getTargetPath("test_lstm_with_initial_bias"))
    }

    @Test
    fun test_bilstm_defaults() {
        Utils.singleTestHelper(getTargetPath("test_bilstm_defaults"))
    }
}
