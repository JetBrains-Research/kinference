package io.kinference.operators.math

import io.kinference.runners.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GeluTest {
    @Test
    fun test_GELU() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources("/gelu/")
    }
}
