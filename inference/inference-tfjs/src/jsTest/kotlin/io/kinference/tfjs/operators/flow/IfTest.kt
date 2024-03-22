package io.kinference.tfjs.operators.flow

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IfTest {
    @Test
    fun test_if() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources("if/")
    }
}
