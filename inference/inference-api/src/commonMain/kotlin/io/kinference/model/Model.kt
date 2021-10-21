package io.kinference.model

import io.kinference.InferenceEngine

interface Model<T> {
    fun predict(input: Map<String, T>, profile: Boolean = false): Map<String, T>

    companion object {
        fun load(bytes: ByteArray, engine: InferenceEngine) = engine.loadModel(bytes)
        //inline fun <reified T> load(bytes: ByteArray, engine: InferenceEngine, adapter: ONNXDataAdapter<T>) = engine.loadModel(bytes, adapter)
    }
}
