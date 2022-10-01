package io.kinference

import io.kinference.data.*
import io.kinference.model.Model
import okio.Path
import okio.Path.Companion.toPath

abstract class BackendInfo(val name: String)

interface InferenceEngine<T : ONNXData<*, *>> {
    val info: BackendInfo

    fun loadData(bytes: ByteArray, type: ONNXDataType): T
    fun loadModel(bytes: ByteArray): Model<T>

    suspend fun loadModel(path: Path): Model<T>
    suspend fun loadData(path: Path, type: ONNXDataType): T

    suspend fun loadModel(path: String): Model<T> = loadModel(path.toPath())
    suspend fun loadData(path: String, type: ONNXDataType): T = loadData(path.toPath(), type)
}
