package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceMapTest {
    private fun getTargetPath(dirName: String) = "sequence_map/$dirName/"

    @Test
    fun test_sequence_map_add_1_sequence_1_tensor() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_1_sequence_1_tensor"))
    }

    @Test
    fun test_sequence_map_add_1_sequence_1_tensor_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_1_sequence_1_tensor_expanded"))
    }

    @Test
    fun test_sequence_map_add_2_sequences() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_2_sequences"))
    }

    @Test
    fun test_sequence_map_add_2_sequences_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_2_sequences_expanded"))
    }

    @Test
    fun test_sequence_map_extract_shapes() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_extract_shapes"))
    }

    @Test
    fun test_sequence_map_extract_shapes_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_extract_shapes_expanded"))
    }

    @Test
    fun test_sequence_map_identity_1_sequence() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence"))
    }

    @Test
    fun test_sequence_map_identity_1_sequence_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_expanded"))
    }

    @Test
    fun test_sequence_map_identity_1_sequence_1_tensor() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_1_tensor"))
    }

    @Test
    fun test_sequence_map_identity_1_sequence_1_tensor_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_1_tensor_expanded"))
    }

    @Test
    fun test_sequence_map_identity_2_sequences() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_2_sequences"))
    }

    @Test
    fun test_sequence_map_identity_2_sequences_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_2_sequences_expanded"))
    }
}

