package io.kinference.operators.seq

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceInsertTest {
    private fun getTargetPath(dirName: String) = "sequence_insert/$dirName/"

    @Test
    fun test_sequence_insert_at_front() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_insert_at_front"))
    }

    @Test
    fun test_sequence_insert_at_back() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_insert_at_back"))
    }
}
