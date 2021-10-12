package io.kinference.ort.model

import ai.onnxruntime.*
import io.kinference.data.ONNXDataAdapter
import io.kinference.model.Model
import io.kinference.ort.data.utils.createORTData
import io.kinference.utils.LoggerFactory

class ORTModel<T>(private val session: OrtSession, private val adapter: ONNXDataAdapter<T>) : Model<T> {
    override fun predict(input: Map<String, T>, profile: Boolean): Map<String, T> {
        if (profile) logger.warning { "Profiling of models running on OnnxRuntime backend is not supported" }

        val onnxInput = input.map { (name, data) -> adapter.toONNXData(name, data) }
        val inputsMap = onnxInput.associate { it.name to it.data as OnnxTensor }
        val outputMap = session.run(inputsMap)
        return outputMap.associate { it.key to adapter.fromONNXData(createORTData(it.key, it.value)) }
    }

    companion object {
        private val logger = LoggerFactory.create("io.kinference.ort.model.ORTModel")
    }
}
