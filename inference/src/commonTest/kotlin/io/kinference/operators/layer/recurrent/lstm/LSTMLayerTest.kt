package io.kinference.operators.layer.recurrent.lstm

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class LSTMLayerTest {
    private fun getTargetPath(dirName: String) = "/lstm/$dirName/"

    @Test
    fun test_LSTM_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_lstm_defaults"))
    }

    @Test
    fun test_LSTM_with_initial_bias()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_lstm_with_initial_bias"))
    }

    @Test
    fun test_LSTM_with_peepholes() = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_lstm_with_peepholes"))
    }

    @Test
    fun test_BiLSTM_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_bilstm_defaults"))
    }

    @Test
    fun test_BiLSTM_with_bias()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_bilstm_with_bias"))
    }
}
