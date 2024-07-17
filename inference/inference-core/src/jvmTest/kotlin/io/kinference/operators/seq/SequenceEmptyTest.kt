package io.kinference.operators.seq

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceEmptyTest {
    private fun getTargetPath(dirName: String) = "sequence_empty/$dirName/"

    @Test
    fun test_sequence_empty() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_empty"))
    }
}
