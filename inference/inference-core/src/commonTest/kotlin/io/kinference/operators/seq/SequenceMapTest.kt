package io.kinference.operators.seq

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceMapTest {
    private fun getTargetPath(dirName: String) = "sequence_map/$dirName/"

    @Test
    fun test_sequence_map_add_1_sequence_1_tensor() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_1_sequence_1_tensor"))
    }

    @Test
    fun test_sequence_map_add_1_sequence_1_tensor_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_1_sequence_1_tensor_expanded"))
    }

    @Test
    fun test_sequence_map_add_2_sequences() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_2_sequences"))
    }

    @Test
    fun test_sequence_map_add_2_sequences_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_2_sequences_expanded"))
    }

    @Test
    fun test_sequence_map_extract_shapes() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_extract_shapes"))
    }

    @Test
    fun test_sequence_map_extract_shapes_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_extract_shapes_expanded"))
    }

    @Test
    fun test_sequence_map_identity_1_sequence() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence"))
    }

    @Test
    fun test_sequence_map_identity_1_sequence_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_expanded"))
    }

    @Test
    fun test_sequence_map_identity_1_sequence_1_tensor() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_1_tensor"))
    }

    @Test
    fun test_sequence_map_identity_1_sequence_1_tensor_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_1_tensor_expanded"))
    }

    @Test
    fun test_sequence_map_identity_2_sequences() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_2_sequences"))
    }

    @Test
    fun test_sequence_map_identity_2_sequences_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_2_sequences_expanded"))
    }
}

