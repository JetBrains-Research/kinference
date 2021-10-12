package io.kinference

import io.kinference.data.*
import io.kinference.model.Model
import io.kinference.utils.KILogger

abstract class TestEngine(private val engine: InferenceEngine) {
    abstract fun checkEquals(expected: ONNXData<*>, actual: ONNXData<*>, delta: Double)
    fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*> = engine.loadData(bytes, type)
    fun loadModel(bytes: ByteArray): Model<ONNXData<*>> = engine.loadModel(bytes, IdAdapter)
}

expect object TestLoggerFactory {
    fun create(name: String): KILogger
}
