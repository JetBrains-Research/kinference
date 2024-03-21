package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class GeluTest {
    @Test
    fun test_GELU() = runTest {
        KIAccuracyRunner.runFromResources("gelu/")
    }
}
