package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ReverseSequenceTest {
    private fun getTargetPath(dirName: String) = "reverse_sequence/$dirName/"

    @Test
    fun test_reverse_sequence_batch() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reversesequence_batch"))
    }

    @Test
    fun test_reverse_sequence_batch_3d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_batch_3d"))
    }

    @Test
    fun test_reverse_sequence_batch_4d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_batch_4d"))
    }

    @Test
    fun test_reverse_sequence_time() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reversesequence_time"))
    }

    @Test
    fun test_reverse_sequence_time_3d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_time_3d"))
    }

    @Test
    fun test_reverse_sequence_time_4d() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_time_4d"))
    }
}
