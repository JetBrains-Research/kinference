package io.kinference.tfjs.operators.seq

import io.kinference.tfjs.runners.TFJSTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SequenceConstructTest {
    private fun getTargetPath(dirName: String) = "sequence_construct/$dirName/"

    @Test
    fun test_sequence_construct() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_sequence_construct"))
    }
}
