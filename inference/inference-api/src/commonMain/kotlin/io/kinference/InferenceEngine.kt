package io.kinference

import io.kinference.data.*
import io.kinference.model.Model

interface InferenceEngine {
    //fun <T> loadModel(bytes: ByteArray, adapter: ONNXDataAdapter<T>): Model<T>
    fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*>
    fun loadModel(bytes: ByteArray): Model<ONNXData<*>>
}
