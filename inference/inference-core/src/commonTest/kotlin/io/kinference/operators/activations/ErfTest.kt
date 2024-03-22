package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ErfTest {
    @Test
    fun test_erf() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources("erf/")
    }
}
