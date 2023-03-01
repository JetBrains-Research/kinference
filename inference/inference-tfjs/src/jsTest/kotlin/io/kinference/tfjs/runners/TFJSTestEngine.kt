package io.kinference.tfjs.runners

import io.kinference.TestEngine
import io.kinference.WithMemory
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.TFJSEngine
import io.kinference.tfjs.utils.TFJSAssertions

object TFJSTestEngine : TestEngine<TFJSData<*>>(TFJSEngine), WithMemory {
    override fun checkEquals(expected: TFJSData<*>, actual: TFJSData<*>, delta: Double) {
        TFJSAssertions.assertEquals(expected, actual, delta)
    }

    override fun memory(): Int = io.kinference.ndarray.core.memory().numTensors

    val TFJSAccuracyRunner = AccuracyRunner(TFJSTestEngine)
    val TFJSPerformanceRunner = PerformanceRunner(TFJSTestEngine)
}
