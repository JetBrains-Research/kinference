package io.kinference.kmath

import io.kinference.data.*
import io.kinference.ort.ORTData
import io.kinference.ort.model.ORTModel
import space.kscience.kmath.nd.NDStructure

class ORTKMathAdapter(model: ORTModel) : ONNXModelAdapter<ORTData<*>>(model) {
    override val adapters: Map<ONNXDataType, ONNXDataAdapter<Any, ORTData<*>>> = mapOf(
        ONNXDataType.ONNX_TENSOR to ORTKMathTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to ORTKMathSequenceAdapter,
        ONNXDataType.ONNX_MAP to ORTKMathMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<Any, ORTData<*>>>

    override fun <T> T.onnxType(): ONNXDataType = when (this) {
        is NDStructure<*> -> ONNXDataType.ONNX_TENSOR
        is List<*> -> ONNXDataType.ONNX_SEQUENCE
        is Map<*, *> -> ONNXDataType.ONNX_MAP
        else -> error("Cannot resolve ONNX data type for ${this!!::class}")
    }
}
