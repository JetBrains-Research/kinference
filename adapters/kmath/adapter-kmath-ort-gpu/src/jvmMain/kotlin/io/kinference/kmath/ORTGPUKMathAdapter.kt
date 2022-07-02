package io.kinference.kmath

import io.kinference.data.*
import io.kinference.ort_gpu.ORTGPUData
import io.kinference.ort_gpu.data.map.ORTGPUMap
import io.kinference.ort_gpu.data.seq.ORTGPUSequence
import io.kinference.ort_gpu.data.tensor.ORTGPUTensor
import io.kinference.ort_gpu.model.ORTGPUModel

class ORTGPUKMathAdapter(model: ORTGPUModel) : ONNXModelAdapter<ORTGPUKMathData<*>, ORTGPUData<*>>(model) {
    override val adapters = mapOf(
        ONNXDataType.ONNX_TENSOR to ORTGPUKMathTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to ORTGPUKMathSequenceAdapter,
        ONNXDataType.ONNX_MAP to ORTGPUKMathMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<ORTGPUKMathData<*>, ORTGPUData<*>>>

    override fun finalizeData(data: Collection<ORTGPUData<*>>) {
        for (element in data) {
            when (element.type) {
                ONNXDataType.ONNX_TENSOR -> (element.data as ORTGPUTensor).data.close()
                ONNXDataType.ONNX_SEQUENCE-> (element.data as ORTGPUSequence).data.close()
                ONNXDataType.ONNX_MAP -> (element.data as ORTGPUMap).data.close()
            }
        }
    }
}
