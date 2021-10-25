package io.kinference.data

import io.kinference.model.Model

abstract class ONNXModelAdapter<SourceType, TargetType : ONNXData<*, *>>(private val model: Model<TargetType>) {
    abstract fun toONNXData(name: String, data: SourceType): TargetType
    abstract fun fromONNXData(data: TargetType): SourceType

    open fun predict(inputs: Map<String, SourceType>, profile: Boolean = false): Map<String, SourceType> {
        val onnxInputs = inputs.mapValues { toONNXData(it.key, it.value) }
        val result = model.predict(onnxInputs, profile)
        return result.mapValues { fromONNXData(it.value) }
    }
}
