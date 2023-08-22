package io.kinference.operators.seq

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

//TODO: Update corresponding operators to enable expanded tests
class SequenceMapTest {
    private fun getTargetPath(dirName: String) = "sequence_map/$dirName/"

    @Test
    fun test_sequence_map_add_1_sequence_1_tensor() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_1_sequence_1_tensor"))
    }

    /*@Test
    fun test_sequence_map_add_1_sequence_1_tensor_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_1_sequence_1_tensor_expanded"))
    }*/

    @Test
    fun test_sequence_map_add_2_sequences() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_2_sequences"))
    }

    /*@Test
    fun test_sequence_map_add_2_sequences_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_add_2_sequences_expanded"))
    }*/

    @Test
    fun test_sequence_map_extract_shapes() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_extract_shapes"))
    }

    /*@Test
    fun test_sequence_map_extract_shapes_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_extract_shapes_expanded"))
    }*/

    @Test
    fun test_sequence_map_identity_1_sequence() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence"))
    }

    /*@Test
    fun test_sequence_map_identity_1_sequence_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_expanded"))
    }*/

    @Test
    fun test_sequence_map_identity_1_sequence_1_tensor() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_1_tensor"))
    }

    /*@Test
    fun test_sequence_map_identity_1_sequence_1_tensor_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_1_sequence_1_tensor_expanded"))
    }*/

    @Test
    fun test_sequence_map_identity_2_sequences() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_2_sequences"))
    }

    /*@Test
    fun test_sequence_map_identity_2_sequences_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_map_identity_2_sequences_expanded"))
    }*/
}

