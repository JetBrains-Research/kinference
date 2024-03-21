package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ErfTest {
    @Test
    fun test_erf() = runTest {
        TFJSAccuracyRunner.runFromResources("erf/")
    }
}
