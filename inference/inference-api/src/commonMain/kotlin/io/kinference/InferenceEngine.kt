package io.kinference

import io.kinference.data.*
import io.kinference.model.Model

abstract class BackendInfo(val name: String)

interface InferenceEngine<T : ONNXData<*, *>> {
    val info: BackendInfo

    fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*, *>
    fun loadModel(bytes: ByteArray): Model<T>

    suspend fun loadDataSuspend(bytes: ByteArray, type: ONNXDataType): ONNXData<*, *> = loadData(bytes, type)
    suspend fun loadModelSuspend(bytes: ByteArray): Model<T> = loadModel(bytes)
}
