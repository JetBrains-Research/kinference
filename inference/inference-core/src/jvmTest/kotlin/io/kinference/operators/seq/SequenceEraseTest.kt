package io.kinference.operators.seq

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceEraseTest {
    private fun getTargetPath(dirName: String) = "sequence_erase/$dirName/"

    @Test
    fun test_sequence_erase_default() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_default"))
    }

    @Test
    fun test_sequence_erase_positive() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_positive"))
    }

    @Test
    fun test_sequence_erase_negative() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_negative"))
    }
}
