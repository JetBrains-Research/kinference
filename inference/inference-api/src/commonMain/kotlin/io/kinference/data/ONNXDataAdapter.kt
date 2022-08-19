package io.kinference.data

import io.kinference.model.Model

interface ONNXDataAdapter<SourceType : BaseONNXData<*>, TargetType : ONNXData<*, *>> {
    fun toONNXData(data: SourceType): TargetType
    fun fromONNXData(data: TargetType): SourceType
}

abstract class ONNXModelAdapter<SourceType : BaseONNXData<*>, TargetType : ONNXData<*, *>>(private val model: Model<TargetType>) {
    protected abstract val adapters: Map<ONNXDataType, ONNXDataAdapter<SourceType, TargetType>>

    open fun predict(inputs: List<SourceType>, profile: Boolean = false): Map<String, SourceType> {
        val onnxInputs = inputs.map { adapters[it.type]!!.toONNXData(it) }
        val onnxResult = model.predict(onnxInputs, profile)
        val result = onnxResult.mapValues { adapters[it.value.type]!!.fromONNXData(it.value) }
        finalizeData(onnxInputs)
        finalizeData(onnxResult.values)
        return result
    }

    protected open fun finalizeData(data: Collection<TargetType>) = Unit
}
