package io.kinference.model

import io.kinference.InferenceEngine
import io.kinference.data.ONNXData

interface Model<T : ONNXData<*, *>> {
    fun predict(input: List<T>, profile: Boolean = false): Map<String, T>
    suspend fun predictSuspend(input: List<T>, profile: Boolean = false): Map<String, T> = predict(input, profile)

    companion object {
        fun <T : ONNXData<*, *>> load(bytes: ByteArray, engine: InferenceEngine<T>) = engine.loadModel(bytes)
    }
}
