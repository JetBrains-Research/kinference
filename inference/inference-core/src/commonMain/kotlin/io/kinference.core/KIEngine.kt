package io.kinference.core

import io.kinference.InferenceEngine
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.model.KIModel
import io.kinference.data.*
import io.kinference.model.Model
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.ArrayFormat
import io.kinference.protobuf.message.*
import okio.Buffer
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
object KIEngine : InferenceEngine {
    private val KI_READER_CONFIG = ProtobufReader.ReaderConfig(tensorFormat = ArrayFormat.TILED)
    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), KI_READER_CONFIG)

    override fun loadModel(bytes: ByteArray): Model<ONNXData<*>> {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return KIModel(modelScheme, IdAdapter)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> KITensor.create(TensorProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_SEQUENCE -> KIONNXSequence.create(SequenceProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_MAP -> KIONNXMap.create(MapProto.decode(protoReader(bytes)))
    }
}

@OptIn(ExperimentalTime::class)
fun <T> KIEngine.loadModel(bytes: ByteArray, adapter: ONNXDataAdapter<T>): Model<ONNXData<*>> {
    val modelScheme = ModelProto.decode(protoReader(bytes))
    return KIModel(modelScheme, IdAdapter)
}
