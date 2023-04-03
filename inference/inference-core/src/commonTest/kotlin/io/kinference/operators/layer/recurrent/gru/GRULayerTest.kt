package io.kinference.operators.layer.recurrent.gru

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GRULayerTest {
    private fun getTargetPath(dirName: String) = "gru/$dirName/"

    @Test
    fun test_GRU_defaults() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gru_defaults"))
    }

    @Test
    fun test_GRU_with_initial_bias() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gru_with_initial_bias"))
    }

    @Test
    fun test_GRU_seq_length() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gru_seq_length"))
    }
}
