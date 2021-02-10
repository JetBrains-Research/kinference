package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ReluTest {
    @Test
    fun test_relu()  = TestRunner.runTest {
        AccuracyRunner.runFromResources("/relu/")
    }
}
