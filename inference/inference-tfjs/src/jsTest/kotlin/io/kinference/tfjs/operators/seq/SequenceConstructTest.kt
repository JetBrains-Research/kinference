package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceConstructTest {
    private fun getTargetPath(dirName: String) = "sequence_construct/$dirName/"

    @Test
    fun test_sequence_construct() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_construct"))
    }
}
