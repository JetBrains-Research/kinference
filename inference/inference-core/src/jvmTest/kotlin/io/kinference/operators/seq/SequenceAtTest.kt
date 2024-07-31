package io.kinference.operators.seq

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceAtTest {
    private fun getTargetPath(dirName: String) = "sequence_at/$dirName/"

    @Test
    fun test_sequence_at_positive() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_at_positive"))
    }

    @Test
    fun test_sequence_at_negative() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_at_negative"))
    }
}
