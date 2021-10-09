package io.kinference.ort.model

import ai.onnxruntime.*
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataAdapter
import io.kinference.model.Model
import io.kinference.ort.data.utils.createORTData

class ORTModel<T>(private val session: OrtSession, private val adapter: ONNXDataAdapter<T, ONNXData<*>>) : Model<T> {
    override fun predict(input: List<T>, profile: Boolean): List<T> {
        val onnxInput = input.map { adapter.toONNXData(it) }
        val inputsMap = onnxInput.associate { it.name to it.data as OnnxTensor }
        val outputMap = session.run(inputsMap)
        return outputMap.map { adapter.fromONNXData(createORTData(it.key, it.value)) }
    }
}
