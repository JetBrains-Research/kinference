package io.kinference.kmath

import io.kinference.core.KIONNXData
import io.kinference.core.model.KIModel
import io.kinference.data.*

class KIKMathModelAdapter(model: KIModel) : ONNXModelAdapter<KIKMathData<*>, KIONNXData<*>>(model) {
    override val adapters = mapOf(
        ONNXDataType.ONNX_TENSOR to KIKMathTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to KIKMathSequenceAdapter,
        ONNXDataType.ONNX_MAP to KIKMathMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<KIKMathData<*>, KIONNXData<*>>>
}
