package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IdentityTest {
    @Test
    fun test_identity()  = TestRunner.runTest {
        AccuracyRunner.runFromResources("/identity/")
    }
}
