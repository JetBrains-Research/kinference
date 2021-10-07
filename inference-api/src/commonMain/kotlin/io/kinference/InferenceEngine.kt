package io.kinference

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model

interface InferenceEngine {
    fun loadModel(bytes: ByteArray): Model
    fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*>
}
