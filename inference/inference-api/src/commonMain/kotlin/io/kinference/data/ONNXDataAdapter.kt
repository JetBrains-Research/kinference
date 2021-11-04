package io.kinference.data

import io.kinference.model.Model

interface ONNXDataAdapter<SourceType, TargetType : ONNXData<*, *>> {
    fun toONNXData(name: String, data: SourceType): TargetType
    fun fromONNXData(data: TargetType): SourceType
}

abstract class ONNXModelAdapter<TargetType : ONNXData<*, *>>(private val model: Model<TargetType>) {
    protected abstract val adapters: Map<ONNXDataType, ONNXDataAdapter<Any, TargetType>>

    protected abstract fun <T> T.onnxType(): ONNXDataType

    open fun predict(inputs: Map<String, Any>, profile: Boolean = false): Map<String, Any> {
        val onnxInputs = inputs.mapValues { adapters[it.value.onnxType()]!!.toONNXData(it.key, it.value) }
        val result = model.predict(onnxInputs, profile)
        return result.mapValues { adapters[it.value.type]!!.fromONNXData(it.value) }
    }
}
