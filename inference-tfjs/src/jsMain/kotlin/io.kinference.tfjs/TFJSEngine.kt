package io.kinference.tfjs

import io.kinference.InferenceEngine
import io.kinference.data.ONNXDataType
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.ArrayFormat
import io.kinference.protobuf.message.*
import io.kinference.tfjs.data.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.model.TFJSModel
import okio.Buffer

object TFJSEngine : InferenceEngine {
    private val KI_READER_CONFIG = ProtobufReader.ReaderConfig(tensorFormat = ArrayFormat.PRIMITIVE)
    private fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), KI_READER_CONFIG)

    override fun loadModel(bytes: ByteArray): TFJSModel {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return TFJSModel(modelScheme)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): TFJSData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> TFJSTensor.create(TensorProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_SEQUENCE -> TFJSSequence.create(SequenceProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_MAP -> TFJSMap.create(MapProto.decode(protoReader(bytes)))
    }
}
