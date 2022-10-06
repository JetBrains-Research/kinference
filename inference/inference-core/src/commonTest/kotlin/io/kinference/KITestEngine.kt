package io.kinference

import io.kinference.core.KIEngine
import io.kinference.core.KIONNXData
import io.kinference.model.ExecutionContext
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.KIAssertions
import kotlinx.coroutines.Dispatchers
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
object KITestEngine : TestEngine<KIONNXData<*>>(KIEngine) {
    override fun checkEquals(expected: KIONNXData<*>, actual: KIONNXData<*>, delta: Double) {
        KIAssertions.assertEquals(expected, actual, delta)
    }

    override fun execContext(): ExecutionContext = ExecutionContext(Dispatchers.Default)

    val KIAccuracyRunner = AccuracyRunner(KITestEngine)
    val KIPerformanceRunner = PerformanceRunner(KITestEngine)
}
