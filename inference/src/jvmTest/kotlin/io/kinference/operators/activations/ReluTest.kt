package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class ReluTest {
    @Test
    fun `test relu`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources("/relu/")
    }
}
