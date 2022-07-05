package io.kinference.kmath

import io.kinference.data.*
import io.kinference.ort.ORTData
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.seq.ORTSequence
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.ort.model.ORTModel

class ORTKMathAdapter(model: ORTModel) : ONNXModelAdapter<ORTKMathData<*>, ORTData<*>>(model) {
    override val adapters = mapOf(
        ONNXDataType.ONNX_TENSOR to ORTKMathTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to ORTKMathSequenceAdapter,
        ONNXDataType.ONNX_MAP to ORTKMathMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<ORTKMathData<*>, ORTData<*>>>

    override fun finalizeData(data: Collection<ORTData<*>>) {
        for (element in data) {
            when (element.type) {
                ONNXDataType.ONNX_TENSOR -> (element as ORTTensor).data.close()
                ONNXDataType.ONNX_SEQUENCE-> (element as ORTSequence).data.close()
                ONNXDataType.ONNX_MAP -> (element as ORTMap).data.close()
            }
        }
    }
}
