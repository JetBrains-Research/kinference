package io.kinference

import io.kinference.core.KIEngine
import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.data.ONNXDataType
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.*

object KITestEngine : TestEngine<KIONNXData<*>>(KIEngine) {
    override fun checkEquals(expected: KIONNXData<*>, actual: KIONNXData<*>, delta: Double) {
        KIAssertions.assertEquals(expected, actual, delta)
    }

    override fun calculateErrors(expected: KIONNXData<*>, actual: KIONNXData<*>): List<Errors.ErrorsData> {
        return KIErrors.calculateErrors(expected, actual)
    }

    override fun getInMemorySize(data: KIONNXData<*>): Int {
        return when(data.type) {
            ONNXDataType.ONNX_TENSOR -> 1
            ONNXDataType.ONNX_SEQUENCE -> (data as KIONNXSequence).data.sumOf { getInMemorySize(it) }
            ONNXDataType.ONNX_MAP -> (data as KIONNXMap).data.values.sumOf { getInMemorySize(it) }
        }
    }

    val KIAccuracyRunner = AccuracyRunner(KITestEngine)
    val KIPerformanceRunner = PerformanceRunner(KITestEngine)
}
