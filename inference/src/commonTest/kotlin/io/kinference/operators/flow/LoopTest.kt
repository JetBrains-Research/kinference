package io.kinference.operators.flow

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LoopTest {
    @Test
    fun test_loop()  = TestRunner.runTest {
        AccuracyRunner.runFromResources("/loop/")
    }
}
