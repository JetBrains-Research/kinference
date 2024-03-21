package io.kinference.operators.quantization

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class TestDynamicQuantizeLSTM {
    private fun getTargetPath(dirName: String) = "dynamic_quantize_lstm/$dirName/"

    @Test
    fun test_LSTM_defaults()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_lstm_defaults"), delta = 1e-2)
    }

    @Test
    fun test_LSTM_with_initial_bias()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_lstm_with_initial_bias"), delta = 1e-2)
    }

    @Test
    fun test_LSTM_with_peepholes() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_lstm_with_peepholes"), delta = 1e-2)
    }

    @Test
    fun test_BiLSTM_defaults()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_bilstm_defaults"), delta = 1e-2)
    }
}
