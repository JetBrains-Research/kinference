package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ReluTest {
    @Test
    fun test_relu() = runTest {
        TFJSAccuracyRunner.runFromResources("relu/")
    }
}
