package io.kinference.operators.seq

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceLengthTest {
    private fun getTargetPath(dirName: String) = "sequence_length/$dirName/"

    @Test
    fun test_sequence_length() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_length"))
    }

    @Test
    fun test_sequence_length_empty() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_length_empty"))
    }
}
