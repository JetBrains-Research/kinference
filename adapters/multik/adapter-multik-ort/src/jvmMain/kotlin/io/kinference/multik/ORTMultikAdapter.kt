package io.kinference.multik

import io.kinference.data.*
import io.kinference.ort.ORTData
import io.kinference.ort.model.ORTModel
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray

class ORTMultikAdapter(model: ORTModel) : ONNXModelAdapter<ORTData<*>>(model) {
    override val adapters: Map<ONNXDataType, ONNXDataAdapter<Any, ORTData<*>>> = mapOf(
        ONNXDataType.ONNX_TENSOR to ORTMultikTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to ORTMultikSequenceAdapter,
        ONNXDataType.ONNX_MAP to ORTMultikMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<Any, ORTData<*>>>

    override fun <T> T.onnxType(): ONNXDataType = when (this) {
        is MultiArray<*, *> -> ONNXDataType.ONNX_TENSOR
        is List<*> -> ONNXDataType.ONNX_SEQUENCE
        is Map<*, *> -> ONNXDataType.ONNX_MAP
        else -> error("Cannot resolve ONNX data type for ${this!!::class}")
    }
}
