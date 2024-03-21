package io.kinference.operators.seq

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceEraseTest {
    private fun getTargetPath(dirName: String) = "sequence_erase/$dirName/"

    @Test
    fun test_sequence_erase_default() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_default"))
    }

    @Test
    fun test_sequence_erase_positive() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_positive"))
    }

    @Test
    fun test_sequence_erase_negative() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_negative"))
    }
}
