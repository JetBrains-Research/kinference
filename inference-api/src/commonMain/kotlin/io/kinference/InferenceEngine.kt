package io.kinference

import io.kinference.data.*
import io.kinference.model.Model

interface InferenceEngine<ModelDataType : ONNXData<*>> {
    fun <T> loadModel(bytes: ByteArray, adapter: ONNXDataAdapter<T, ModelDataType>): Model<T>
    fun loadData(bytes: ByteArray, type: ONNXDataType): ModelDataType
}

fun <T : ONNXData<*>, E : InferenceEngine<T>> E.loadModel(bytes: ByteArray): Model<T> = loadModel(bytes, ONNXDataAdapter.idAdapter())
