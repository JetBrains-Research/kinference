package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceEraseTest {
    private fun getTargetPath(dirName: String) = "sequence_erase/$dirName/"

    @Test
    fun test_sequence_erase_default() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_default"))
    }

    @Test
    fun test_sequence_erase_positive() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_positive"))
    }

    @Test
    fun test_sequence_erase_negative() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_erase_negative"))
    }
}
