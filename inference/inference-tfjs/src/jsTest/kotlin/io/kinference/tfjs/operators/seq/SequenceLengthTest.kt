package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceLengthTest {
    private fun getTargetPath(dirName: String) = "sequence_length/$dirName/"

    @Test
    fun test_sequence_length() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_length"))
    }

    @Test
    fun test_sequence_length_empty() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_length_empty"))
    }
}
