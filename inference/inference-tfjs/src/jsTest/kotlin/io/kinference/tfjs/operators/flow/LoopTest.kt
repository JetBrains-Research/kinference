package io.kinference.tfjs.operators.flow

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class LoopTest {
    @Test
    fun test_loop() = runTest {
        TFJSAccuracyRunner.runFromResources("loop/")
    }
}
