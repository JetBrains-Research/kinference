package io.kinference.operators.operations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ReverseSequenceTest {
    private fun getTargetPath(dirName: String) = "reverse_sequence/$dirName/"

    @Test
    fun test_reverse_sequence_batch() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reversesequence_batch"))
    }

    @Test
    fun test_reverse_sequence_string_batch() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_string_batch"))
    }

    @Test
    fun test_reverse_sequence_batch_3d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_batch_3d"))
    }

    @Test
    fun test_reverse_sequence_string_batch_3d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_string_batch_3d"))
    }

    @Test
    fun test_reverse_sequence_batch_4d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_batch_4d"))
    }

    @Test
    fun test_reverse_sequence_time() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reversesequence_time"))
    }

    @Test
    fun test_reverse_sequence_string_time() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_string_time"))
    }

    @Test
    fun test_reverse_sequence_time_3d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_time_3d"))
    }

    @Test
    fun test_reverse_sequence_string_time_3d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_string_time_3d"))
    }

    @Test
    fun test_reverse_sequence_time_4d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reverse_sequence_time_4d"))
    }
}
