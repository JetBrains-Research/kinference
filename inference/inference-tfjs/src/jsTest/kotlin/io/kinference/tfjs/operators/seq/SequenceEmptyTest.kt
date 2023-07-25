package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceEmptyTest {
    private fun getTargetPath(dirName: String) = "sequence_empty/$dirName/"

    @Test
    fun test_sequence_empty() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_empty"))
    }
}
