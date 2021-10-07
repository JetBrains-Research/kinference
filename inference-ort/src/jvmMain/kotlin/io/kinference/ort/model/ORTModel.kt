package io.kinference.ort.model

import ai.onnxruntime.*
import io.kinference.data.ONNXData
import io.kinference.ort.data.ORTData
import io.kinference.model.Model

class ORTModel(private val session: OrtSession) : Model {
    override fun predict(input: List<ONNXData<*>>, profile: Boolean): List<ONNXData<*>> {
        val inputsMap = input.associate { it.name to it.data as OnnxTensor }
        val outputMap = session.run(inputsMap)
        return outputMap.map { ORTData(it.key, it.value) }
    }
}
