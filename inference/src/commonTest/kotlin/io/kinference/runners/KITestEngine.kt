package io.kinference.runners

import io.kinference.TestEngine
import io.kinference.core.KIEngine
import io.kinference.core.data.KIONNXData
import io.kinference.utils.KIAssertions
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
object KITestEngine : TestEngine<KIONNXData<*>>(KIEngine) {
    override fun checkEquals(expected: KIONNXData<*>, actual: KIONNXData<*>, delta: Double) {
        KIAssertions.assertEquals(expected, actual, delta)
    }

    val KIAccuracyRunner = AccuracyRunner(KITestEngine)
}
