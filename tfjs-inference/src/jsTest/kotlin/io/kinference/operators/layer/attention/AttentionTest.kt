package io.kinference.operators.layer.attention

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlinx.coroutines.delay
import kotlin.test.Test
import kotlin.time.ExperimentalTime

class AttentionTestJS {
    private fun getTargetPath(dirName: String) = "/attention/$dirName/"

    @Test
    fun test_unidirectional_multi_head_masked_attention()  = TestRunner.runTest {
//        delay(15000)
        AccuracyRunner.runFromResources(getTargetPath("test_unidirectional_masked_multi_head"))
    }
}
