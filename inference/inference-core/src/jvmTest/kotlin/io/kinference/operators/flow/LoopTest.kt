package io.kinference.operators.flow

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LoopTest {
    @Test
    fun test_loop() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources("loop/")
    }
}
