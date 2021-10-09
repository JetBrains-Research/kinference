package io.kinference

import io.kinference.data.*
import io.kinference.model.Model

abstract class TestEngine<TestDataType : ONNXData<*>>(private val engine: InferenceEngine<TestDataType>) {
    abstract fun checkEquals(expected: TestDataType, actual: TestDataType, delta: Double)
    fun loadData(bytes: ByteArray, type: ONNXDataType): TestDataType = engine.loadData(bytes, type)
    fun loadModel(bytes: ByteArray): Model<TestDataType> = engine.loadModel(bytes, ONNXDataAdapter.idAdapter())
}
