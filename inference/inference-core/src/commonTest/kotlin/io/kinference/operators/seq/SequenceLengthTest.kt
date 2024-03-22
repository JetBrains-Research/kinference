package io.kinference.operators.seq

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceLengthTest {
    private fun getTargetPath(dirName: String) = "sequence_length/$dirName/"

    @Test
    fun test_sequence_length() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_length"))
    }

    @Test
    fun test_sequence_length_empty() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_length_empty"))
    }
}
