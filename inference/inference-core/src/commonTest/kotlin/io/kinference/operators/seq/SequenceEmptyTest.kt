package io.kinference.operators.seq

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceEmptyTest {
    private fun getTargetPath(dirName: String) = "sequence_empty/$dirName/"

    @Test
    fun test_sequence_empty() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_empty"))
    }
}
