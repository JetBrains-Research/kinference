package io.kinference.model

import io.kinference.InferenceEngine
import io.kinference.data.ONNXData

interface Model<T : ONNXData<*, *>> {
    fun predict(input: List<T>, profile: Boolean = false, executionContext: ExecutionContext? = null): Map<String, T>

    companion object {
        fun <T : ONNXData<*, *>> load(bytes: ByteArray, engine: InferenceEngine<T>) = engine.loadModel(bytes)
    }
}
