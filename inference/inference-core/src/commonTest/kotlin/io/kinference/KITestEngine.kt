package io.kinference

import io.kinference.core.KIEngine
import io.kinference.core.KIONNXData
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.KIAssertions
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
object KITestEngine : TestEngine<KIONNXData<*>>(KIEngine) {
    override fun checkEquals(expected: KIONNXData<*>, actual: KIONNXData<*>, delta: Double) {
        KIAssertions.assertEquals(expected, actual, delta)
    }

    override fun postprocessData(data: KIONNXData<*>) = Unit

    val KIAccuracyRunner = AccuracyRunner(KITestEngine)
    val KIPerformanceRunner = PerformanceRunner(KITestEngine)
}
