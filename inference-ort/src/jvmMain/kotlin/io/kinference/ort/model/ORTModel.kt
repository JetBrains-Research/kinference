package io.kinference.ort.model

import ai.onnxruntime.*
import io.kinference.data.ONNXDataAdapter
import io.kinference.ort.data.ORTData
import io.kinference.model.Model

class ORTModel<T>(private val session: OrtSession, private val adapter: ONNXDataAdapter<T, ORTData>) : Model<T> {
    override fun predict(input: List<T>, profile: Boolean): List<T> {
        val onnxInput = input.map { adapter.toONNXData(it) }
        val inputsMap = onnxInput.associate { it.name to it.data as OnnxTensor }
        val outputMap = session.run(inputsMap)
        return outputMap.map { adapter.fromONNXData(ORTData(it.key, it.value)) }
    }
}
