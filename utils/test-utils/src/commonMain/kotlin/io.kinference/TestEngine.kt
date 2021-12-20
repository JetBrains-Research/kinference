package io.kinference

import io.kinference.data.*
import io.kinference.model.Model
import io.kinference.utils.KILogger

abstract class TestEngine<T : ONNXData<*, *>>(private val engine: InferenceEngine<T>) {
    abstract fun checkEquals(expected: T, actual: T, delta: Double)
    abstract fun postprocessData(data: T)
    suspend fun loadData(bytes: ByteArray, type: ONNXDataType): T = engine.loadDataSuspend(bytes, type) as T
    suspend fun loadModel(bytes: ByteArray): Model<T> = engine.loadModelSuspend(bytes)
}

expect object TestLoggerFactory {
    fun create(name: String): KILogger
}
