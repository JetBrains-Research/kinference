package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceInsertTest {
    private fun getTargetPath(dirName: String) = "sequence_insert/$dirName/"

    @Test
    fun test_sequence_insert_at_front() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_insert_at_front"))
    }

    @Test
    fun test_sequence_insert_at_back() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_insert_at_back"))
    }
}
