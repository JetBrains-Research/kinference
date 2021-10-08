package io.kinference.tfjs.runners

import io.kinference.TestEngine
import io.kinference.data.ONNXData
import io.kinference.runners.AccuracyRunner
import io.kinference.tfjs.TFJSEngine
import io.kinference.tfjs.data.TFJSData
import io.kinference.tfjs.utils.TFJSAssertions

object TFJSTestEngine : TestEngine(TFJSEngine) {
    override fun checkEquals(expected: ONNXData<*>, actual: ONNXData<*>, delta: Double) {
        TFJSAssertions.assertEquals(expected as TFJSData<*>, actual as TFJSData<*>, delta)
    }

    val TFJSAccuracyRunner = AccuracyRunner(TFJSTestEngine)
}
