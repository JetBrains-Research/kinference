package io.kinference.tfjs.operators.flow

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class IfTest {
    @Test
    fun test_if() = runTest {
        TFJSAccuracyRunner.runFromResources("if/")
    }
}
