package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceAtTest {
    private fun getTargetPath(dirName: String) = "sequence_at/$dirName/"

    @Test
    fun test_sequence_at_positive() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_at_positive"))
    }

    @Test
    fun test_sequence_at_negative() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_at_negative"))
    }
}
