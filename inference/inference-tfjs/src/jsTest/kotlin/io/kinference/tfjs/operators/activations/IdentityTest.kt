package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class IdentityTest {
    @Test
    fun test_identity() = runTest {
        TFJSAccuracyRunner.runFromResources("identity/")
    }
}
