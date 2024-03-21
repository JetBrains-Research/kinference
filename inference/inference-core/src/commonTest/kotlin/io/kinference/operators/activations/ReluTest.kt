package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ReluTest {
    @Test
    fun test_relu() = runTest {
        KIAccuracyRunner.runFromResources("relu/")
    }
}
