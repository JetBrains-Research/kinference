package io.kinference.operators.flow

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IfTest {

    @Test
    fun test_if()  = TestRunner.runTest {
        AccuracyRunner.runFromResources("/if/")
    }
}
