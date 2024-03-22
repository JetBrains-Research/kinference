package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ReluTest {
    @Test
    fun test_relu() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources("relu/")
    }
}
