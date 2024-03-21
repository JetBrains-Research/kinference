package io.kinference.operators.seq

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceInsertTest {
    private fun getTargetPath(dirName: String) = "sequence_insert/$dirName/"

    @Test
    fun test_sequence_insert_at_front() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_insert_at_front"))
    }

    @Test
    fun test_sequence_insert_at_back() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_insert_at_back"))
    }
}
