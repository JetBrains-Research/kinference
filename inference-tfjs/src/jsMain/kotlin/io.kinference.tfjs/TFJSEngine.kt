package io.kinference.tfjs

import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.protobuf.message.*
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.model.TFJSModel

object TFJSEngine : InferenceEngine {
    override fun loadModel(bytes: ByteArray): Model {
        val modelScheme = ModelProto.decode(bytes)
        return TFJSModel(modelScheme)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> TFJSTensor.create(TensorProto.decode(bytes))
        ONNXDataType.ONNX_SEQUENCE -> TFJSSequence.create(SequenceProto.decode(bytes))
        ONNXDataType.ONNX_MAP -> TFJSMap.create(MapProto.decode(bytes))
    }
}
