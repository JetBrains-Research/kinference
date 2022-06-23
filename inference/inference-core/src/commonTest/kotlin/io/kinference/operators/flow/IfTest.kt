package io.kinference.operators.flow

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IfTest {

    @Test
    fun test_if() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources("if/")
    }
}
