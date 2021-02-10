package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ErfTest {
    @Test
    fun test_erf()  = TestRunner.runTest {
        AccuracyRunner.runFromResources("/erf/")
    }
}
