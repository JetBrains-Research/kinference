package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceAtTest {
    private fun getTargetPath(dirName: String) = "sequence_at/$dirName/"

    @Test
    fun test_sequence_at_positive() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_at_positive"))
    }

    @Test
    fun test_sequence_at_negative() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_at_negative"))
    }
}
