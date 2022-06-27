package io.kinference.operators.quantization

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TestDynamicQuantizeLSTM {
    private fun getTargetPath(dirName: String) = "dynamic_quantize_lstm/$dirName/"

    @Test
    fun test_LSTM_defaults()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_lstm_defaults"), delta = 1e-2)
    }

    @Test
    fun test_LSTM_with_initial_bias()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_lstm_with_initial_bias"), delta = 1e-2)
    }

    @Test
    fun test_LSTM_with_peepholes() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_lstm_with_peepholes"), delta = 1e-2)
    }

    @Test
    fun test_BiLSTM_defaults()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_bilstm_defaults"), delta = 1e-2)
    }
}
