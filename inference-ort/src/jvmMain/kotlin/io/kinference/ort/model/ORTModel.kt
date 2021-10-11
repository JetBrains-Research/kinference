package io.kinference.ort.model

import ai.onnxruntime.*
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataAdapter
import io.kinference.model.Model
import io.kinference.ort.data.utils.createORTData

class ORTModel<T>(private val session: OrtSession, private val adapter: ONNXDataAdapter<T>) : Model<T> {
    override fun predict(input: Map<String, T>, profile: Boolean): Map<String, T> {
        val onnxInput = input.map { (name, data) -> adapter.toONNXData(name, data) }
        val inputsMap = onnxInput.associate { it.name to it.data as OnnxTensor }
        val outputMap = session.run(inputsMap)
        return outputMap.associate { it.key to adapter.fromONNXData(createORTData(it.key, it.value)) }
    }
}
