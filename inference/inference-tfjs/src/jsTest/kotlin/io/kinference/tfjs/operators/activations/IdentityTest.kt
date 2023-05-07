package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IdentityTest {
    @Test
    fun test_identity() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources("identity/")
    }
}
