package io.kinference.multik

import io.kinference.data.*
import io.kinference.ort.ORTData
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.seq.ORTSequence
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.ort.model.ORTModel

class ORTMultikAdapter(model: ORTModel) : ONNXModelAdapter<ORTMultikData<*>, ORTData<*>>(model) {
    override val adapters = mapOf(
        ONNXDataType.ONNX_TENSOR to ORTMultikTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to ORTMultikSequenceAdapter,
        ONNXDataType.ONNX_MAP to ORTMultikMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<ORTMultikData<*>, ORTData<*>>>
}
