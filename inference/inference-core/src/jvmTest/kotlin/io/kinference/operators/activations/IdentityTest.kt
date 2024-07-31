package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IdentityTest {
    @Test
    fun test_identity() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources("identity/")
    }
}
