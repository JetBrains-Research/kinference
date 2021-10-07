package io.kinference

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model

abstract class TestEngine(private val engine: InferenceEngine) {
    abstract fun checkEquals(expected: ONNXData<*>, actual: ONNXData<*>, delta: Double)
    fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*> = engine.loadData(bytes, type)
    fun loadModel(bytes: ByteArray): Model = engine.loadModel(bytes)
}
