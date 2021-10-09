package io.kinference.model

import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataAdapter
import io.kinference.loadModel

interface Model<T> {
    fun predict(input: List<T>, profile: Boolean = false): List<T>

    companion object {
        fun <T : ONNXData<*>> load(bytes: ByteArray, engine: InferenceEngine<T>) = engine.loadModel(bytes)
        fun <T, V : ONNXData<*>> load(bytes: ByteArray, engine: InferenceEngine<V>, adapter: ONNXDataAdapter<T, V>) = engine.loadModel(bytes, adapter)
    }
}
