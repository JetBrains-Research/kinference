package io.kinference

import io.kinference.data.*
import io.kinference.model.Model
import io.kinference.utils.KILogger

abstract class TestEngine<T : ONNXData<*, *>>(private val engine: InferenceEngine<T>) {
    abstract fun checkEquals(expected: T, actual: T, delta: Double)
    abstract fun postprocessData(data: T)
    fun loadData(bytes: ByteArray, type: ONNXDataType): T = engine.loadData(bytes, type) as T
    fun loadModel(bytes: ByteArray): Model<T> = engine.loadModel(bytes)
}

expect object TestLoggerFactory {
    fun create(name: String): KILogger
}
