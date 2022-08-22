package io.kinference.tfjs.operators.flow

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class LoopTest {
    @Test
    fun test_loop() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources("loop/")
    }
}
