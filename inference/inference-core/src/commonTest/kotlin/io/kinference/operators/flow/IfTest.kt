package io.kinference.operators.flow

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class IfTest {

    @Test
    fun test_if() = runTest {
        KIAccuracyRunner.runFromResources("if/")
    }
}
