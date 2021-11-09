package io.kinference.multik

import io.kinference.core.KIONNXData
import io.kinference.core.model.KIModel
import io.kinference.data.*

class KIMultikAdapter(model: KIModel) : ONNXModelAdapter<KIMultikData<*>, KIONNXData<*>>(model) {
    override val adapters = mapOf(
        ONNXDataType.ONNX_TENSOR to KIMultikTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to KIMultikSequenceAdapter,
        ONNXDataType.ONNX_MAP to KIMultikMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<KIMultikData<*>, KIONNXData<*>>>
}

