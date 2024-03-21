package io.kinference.operators.flow

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LoopTest {
    @Test
    fun test_loop() = runTest {
        KIAccuracyRunner.runFromResources("loop/")
    }
}
