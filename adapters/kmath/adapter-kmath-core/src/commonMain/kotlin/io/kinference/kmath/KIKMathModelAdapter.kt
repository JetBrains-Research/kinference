package io.kinference.kmath

import io.kinference.core.KIONNXData
import io.kinference.core.model.KIModel
import io.kinference.data.*
import space.kscience.kmath.nd.NDStructure

class KIKMathModelAdapter(model: KIModel) : ONNXModelAdapter<KIONNXData<*>>(model) {
    override val adapters: Map<ONNXDataType, ONNXDataAdapter<Any, KIONNXData<*>>> = mapOf(
        ONNXDataType.ONNX_TENSOR to KIKMathTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to KIKMathSequenceAdapter,
        ONNXDataType.ONNX_MAP to KIKMathMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<Any, KIONNXData<*>>>

    override fun <T> T.onnxType(): ONNXDataType = when (this) {
        is NDStructure<*> -> ONNXDataType.ONNX_TENSOR
        is List<*> -> ONNXDataType.ONNX_SEQUENCE
        is Map<*, *> -> ONNXDataType.ONNX_MAP
        else -> error("Cannot resolve ONNX data type for ${this!!::class}")
    }
}
