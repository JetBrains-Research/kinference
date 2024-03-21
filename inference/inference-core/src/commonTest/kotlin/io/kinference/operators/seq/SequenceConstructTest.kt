package io.kinference.operators.seq

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceConstructTest {
    private fun getTargetPath(dirName: String) = "sequence_construct/$dirName/"

    @Test
    fun test_sequence_construct() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sequence_construct"))
    }
}
