package io.kinference.model

import io.kinference.InferenceEngine
import io.kinference.data.ONNXData

interface Model {
    fun predict(input: List<ONNXData<*>>, profile: Boolean = false): List<ONNXData<*>>

    companion object {
        fun load(bytes: ByteArray, engine: InferenceEngine) = engine.loadModel(bytes)
    }
}
