package io.kinference

import io.kinference.data.*
import io.kinference.model.Model
import okio.Path
import okio.Path.Companion.toPath

abstract class BackendInfo(val name: String)

interface InferenceEngine<T : ONNXData<*, *>> {
    val info: BackendInfo

    fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*, *>
    fun loadModel(bytes: ByteArray, optimize: Boolean = false): Model<T>

    suspend fun loadModel(path: Path, optimize: Boolean = false): Model<T>
    suspend fun loadData(path: Path, type: ONNXDataType): ONNXData<*, *>

    suspend fun loadModel(path: String, optimize: Boolean = false): Model<T> = loadModel(path.toPath(), optimize)
    suspend fun loadData(path: String, type: ONNXDataType): ONNXData<*, *> = loadData(path.toPath(), type)
}
