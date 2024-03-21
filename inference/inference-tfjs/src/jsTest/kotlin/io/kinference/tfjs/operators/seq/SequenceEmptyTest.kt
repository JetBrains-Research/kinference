package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SequenceEmptyTest {
    private fun getTargetPath(dirName: String) = "sequence_empty/$dirName/"

    @Test
    fun test_sequence_empty() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_empty"))
    }
}
