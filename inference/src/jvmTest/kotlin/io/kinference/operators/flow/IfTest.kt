package io.kinference.operators.flow

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class IfTest {

    @Test
    fun `test if`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources("/if/")
    }
}
