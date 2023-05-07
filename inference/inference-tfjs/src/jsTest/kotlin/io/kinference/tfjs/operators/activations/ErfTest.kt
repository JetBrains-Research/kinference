package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ErfTest {
    @Test
    fun test_erf() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources("erf/")
    }
}
