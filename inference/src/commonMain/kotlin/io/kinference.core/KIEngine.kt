package io.kinference.core

import io.kinference.InferenceEngine
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensors.KITensor
import io.kinference.core.model.KIModel
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.protobuf.message.*
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
object KIEngine : InferenceEngine {
    override fun loadModel(bytes: ByteArray): Model {
        val modelScheme = ModelProto.decode(bytes)
        return KIModel(modelScheme)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> KITensor.create(TensorProto.decode(bytes))
        ONNXDataType.ONNX_SEQUENCE -> KIONNXSequence.create(SequenceProto.decode(bytes))
        ONNXDataType.ONNX_MAP -> KIONNXMap.create(MapProto.decode(bytes))
    }
}
