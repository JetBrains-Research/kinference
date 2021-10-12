package io.kinference.operators.layer.attention

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

class AttentionTest {
    private fun getTargetPath(dirName: String) = "/attention/$dirName/"

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_unidirectional_multi_head_masked_attention() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_unidirectional_masked_multi_head"))
    }
}
