package io.kinference.core

import io.kinference.*
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.model.KIModel
import io.kinference.core.optimizer.rules.DequantizeMatMulInteger
import io.kinference.core.optimizer.rules.DequantizeQAttention
import io.kinference.data.*
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.ArrayFormat
import io.kinference.protobuf.message.*
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path
import kotlin.time.ExperimentalTime

typealias KIONNXData<T> = ONNXData<T, CoreBackend>

object CoreBackend : BackendInfo(name = "KInference Core CPU Backend")

@OptIn(ExperimentalTime::class)
object KIEngine : InferenceEngine<KIONNXData<*>> {
    override val info: BackendInfo = CoreBackend

    private val KI_READER_CONFIG = ProtobufReader.ReaderConfig(tensorFormat = ArrayFormat.TILED)
    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), KI_READER_CONFIG)

    override fun loadModel(bytes: ByteArray, optimize: Boolean): KIModel {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return KIModel(modelScheme, optimize)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): KIONNXData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> KITensor.create(TensorProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_SEQUENCE -> KIONNXSequence.create(SequenceProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_MAP -> KIONNXMap.create(MapProto.decode(protoReader(bytes)))
    }

    val optimizerRules = setOf(DequantizeMatMulInteger, DequantizeQAttention)
    override suspend fun loadData(path: Path, type: ONNXDataType) = loadData(CommonDataLoader.bytes(path), type)
    override suspend fun loadModel(path: Path, optimize: Boolean) = loadModel(CommonDataLoader.bytes(path), optimize)
}
