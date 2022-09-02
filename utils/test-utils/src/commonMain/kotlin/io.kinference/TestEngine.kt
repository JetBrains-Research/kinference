package io.kinference

import io.kinference.data.*
import io.kinference.model.Model
import io.kinference.utils.KILogger
import okio.Path

abstract class TestEngine<T : ONNXData<*, *>>(private val engine: InferenceEngine<T>) {
    abstract fun checkEquals(expected: T, actual: T, delta: Double)
    fun loadData(bytes: ByteArray, type: ONNXDataType): T = engine.loadData(bytes, type) as T
    fun loadModel(bytes: ByteArray): Model<T> = engine.loadModel(bytes)

    suspend fun loadModel(path: Path): Model<T> = engine.loadModel(path)
    suspend fun loadData(path: Path, type: ONNXDataType): T = engine.loadData(path, type) as T
}

expect object TestLoggerFactory {
    fun create(name: String): KILogger
}
