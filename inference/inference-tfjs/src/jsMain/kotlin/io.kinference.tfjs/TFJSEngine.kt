package io.kinference.tfjs

import io.kinference.BackendInfo
import io.kinference.InferenceEngine
import io.kinference.data.*
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.ArrayFormat
import io.kinference.protobuf.message.*
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.model.TFJSModel
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path

typealias TFJSData<T> = ONNXData<T, TFJSBackend>

object TFJSBackend : BackendInfo(name = "TensorFlow for JS")

object TFJSEngine : InferenceEngine<TFJSData<*>> {
    override val info: BackendInfo
        get() = TFJSBackend

    private val TFJS_READER_CONFIG = ProtobufReader.ReaderConfig(tensorFormat = ArrayFormat.PRIMITIVE)
    private fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), TFJS_READER_CONFIG)

    override fun loadModel(bytes: ByteArray): TFJSModel {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return TFJSModel(modelScheme)
    }

    override suspend fun loadModel(path: Path) = loadModel(CommonDataLoader.bytes(path))

    override fun loadData(bytes: ByteArray, type: ONNXDataType): TFJSData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> TFJSTensor.create(TensorProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_SEQUENCE -> TFJSSequence.create(SequenceProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_MAP -> TFJSMap.create(MapProto.decode(protoReader(bytes)))
    }

    override suspend fun loadData(path: Path, type: ONNXDataType) = loadData(CommonDataLoader.bytes(path), type)
}
