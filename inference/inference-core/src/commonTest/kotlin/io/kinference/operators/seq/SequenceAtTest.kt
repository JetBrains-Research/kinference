package io.kinference.operators.seq

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceAtTest {
    private fun getTargetPath(dirName: String) = "sequence_at/$dirName/"

    @Test
    fun test_sequence_at_positive() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_at_positive"))
    }

    @Test
    fun test_sequence_at_negative() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_at_negative"))
    }
}
