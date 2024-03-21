package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class IdentityTest {
    @Test
    fun test_identity() = runTest {
        KIAccuracyRunner.runFromResources("identity/")
    }
}
