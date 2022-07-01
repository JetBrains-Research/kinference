package io.kinference.kmath

import io.kinference.data.*
import io.kinference.ort_gpu.ORTGPUData
import io.kinference.ort_gpu.model.ORTGPUModel

class ORTGPUKMathAdapter(model: ORTGPUModel) : ONNXModelAdapter<ORTGPUKMathData<*>, ORTGPUData<*>>(model) {
    override val adapters = mapOf(
        ONNXDataType.ONNX_TENSOR to ORTGPUKMathTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to ORTGPUKMathSequenceAdapter,
        ONNXDataType.ONNX_MAP to ORTGPUKMathMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<ORTGPUKMathData<*>, ORTGPUData<*>>>
}
