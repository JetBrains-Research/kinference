package io.kinference

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.utils.Errors
import io.kinference.utils.KILogger
import okio.Path

abstract class TestEngine<T : ONNXData<*, *>>(private val engine: InferenceEngine<T>) {
    abstract fun checkEquals(expected: T, actual: T, delta: Double)
    abstract fun getInMemorySize(data: T): Int
    abstract fun calculateErrors(expected: T, actual: T): List<Errors.ErrorsData>

    suspend fun loadData(bytes: ByteArray, type: ONNXDataType): T = engine.loadData(bytes, type)
    suspend fun loadModel(bytes: ByteArray): Model<T> = engine.loadModel(bytes)

    suspend fun loadModel(path: Path): Model<T> = engine.loadModel(path)
    suspend fun loadData(path: Path, type: ONNXDataType): T = engine.loadData(path, type)
}

interface MemoryProfileable {
    fun allocatedMemory(): Int
}

interface Cacheable {
    fun clearCache()
}

expect object TestLoggerFactory {
    fun create(name: String): KILogger
}
