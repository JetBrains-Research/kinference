package io.kinference.model

import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataAdapter
import io.kinference.loadModel

interface Model<T> {
    fun predict(input: List<T>, profile: Boolean = false): List<T>

    companion object {
        fun load(bytes: ByteArray, engine: InferenceEngine) = engine.loadModel(bytes)
        fun <T> load(bytes: ByteArray, engine: InferenceEngine, adapter: ONNXDataAdapter<T, ONNXData<*>>) = engine.loadModel(bytes, adapter)
    }
}
