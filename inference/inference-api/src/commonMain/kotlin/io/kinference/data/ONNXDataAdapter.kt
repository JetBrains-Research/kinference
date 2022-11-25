package io.kinference.data

import io.kinference.model.Model

/**
 * Interface for ONNX data adapters.
 * It provides data conversion methods from [SourceType] to [TargetType] and back.
 *
 * @param SourceType wrapper class type for source data. Source data doesn't necessarily have to be supported by KInference backends.
 * @param TargetType wrapper class type for target data. Target data must be supported by one of the KInference backends.
 */
interface ONNXDataAdapter<SourceType : BaseONNXData<*>, TargetType : ONNXData<*, *>> {
    fun toONNXData(data: SourceType): TargetType
    fun fromONNXData(data: TargetType): SourceType
}

/**
 * Base class defining KInference model adapter.
 * Model adapter enables models to run on specified external data types utilizing data adapters
 * to convert provided inputs to model-appropriate format.
 *
 * @param SourceType wrapper class type for external data. If there is more than one type, it should be the type of wrappers' superclass.
 * @param TargetType data type required by model.
 * @property model model instance to run.
 */
abstract class ONNXModelAdapter<SourceType : BaseONNXData<*>, TargetType : ONNXData<*, *>>(private val model: Model<TargetType>) {
    /**
     * Set of data adapters for each ONNX type.
     */
    protected abstract val adapters: Map<ONNXDataType, ONNXDataAdapter<SourceType, TargetType>>

    /**
     * Runs model pass on data of [SourceType]. First, it converts inputs to [TargetType],
     * then runs the model, and, finally, converts model results back to [SourceType] using provided adapters.
     */
    open fun predict(inputs: List<SourceType>, profile: Boolean = false): Map<String, SourceType> {
        val onnxInputs = inputs.map { adapters[it.type]!!.toONNXData(it) }
        val onnxResult = model.predict(onnxInputs, profile)
        val result = onnxResult.mapValues { adapters[it.value.type]!!.fromONNXData(it.value) }
        onnxInputs.forEach { it.close() }
        onnxResult.values.forEach { it.close() }
        return result
    }
}
