package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ReluTest {
    @Test
    fun test_relu() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources("/relu/")
    }
}
