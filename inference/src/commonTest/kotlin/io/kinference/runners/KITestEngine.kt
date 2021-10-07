package io.kinference.runners

import io.kinference.TestEngine
import io.kinference.core.KIEngine
import io.kinference.core.data.KIONNXData
import io.kinference.data.ONNXData
import io.kinference.utils.KIAssertions
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
object KITestEngine : TestEngine(KIEngine) {
    override fun checkEquals(expected: ONNXData<*>, actual: ONNXData<*>, delta: Double) {
        KIAssertions.assertEquals(expected as KIONNXData<*>, actual as KIONNXData<*>, delta)
    }

    val KIAccuracyRunner = AccuracyRunner(KITestEngine)
}
