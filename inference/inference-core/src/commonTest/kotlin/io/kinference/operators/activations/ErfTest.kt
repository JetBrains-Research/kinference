package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ErfTest {
    @Test
    fun test_erf() = runTest {
        KIAccuracyRunner.runFromResources("erf/")
    }
}
