package io.kinference.core

import io.kinference.BackendInfo
import io.kinference.OptimizableEngine
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.model.KIModel
import io.kinference.core.optimizer.rules.DequantizeQAttention
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.optimizer.GraphOptimizer
import io.kinference.protobuf.*
import io.kinference.protobuf.message.*
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path
import okio.Path.Companion.toPath

typealias KIONNXData<T> = ONNXData<T, CoreBackend>

object CoreBackend : BackendInfo(name = "KInference Core CPU Backend")

object KIEngine : OptimizableEngine<KIONNXData<*>> {
    override val info: BackendInfo = CoreBackend

    private val KI_READER_CONFIG = ProtobufReader.ReaderConfig(tensorDecoder = TiledTensorDecoder)
    private val defaultOptRules = listOf(DequantizeQAttention)
    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), KI_READER_CONFIG)

    override fun loadModel(bytes: ByteArray, optimize: Boolean): KIModel {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        val model = KIModel(modelScheme)
        return if (optimize) {
            val newGraph = GraphOptimizer(model.graph).run(defaultOptRules) as KIGraph
            KIModel(model.name, model.opSet, newGraph)
        } else {
            model
        }
    }

    override fun loadModel(bytes: ByteArray): KIModel = loadModel(bytes, optimize = false)

    override suspend fun loadModel(path: Path, optimize: Boolean): KIModel {
        return loadModel(CommonDataLoader.bytes(path), optimize)
    }

    override suspend fun loadModel(path: Path): KIModel = loadModel(path, optimize = false)

    override suspend fun loadModel(path: String, optimize: Boolean): KIModel {
        return loadModel(CommonDataLoader.bytes(path.toPath()), optimize)
    }

    override suspend fun loadModel(path: String): KIModel = loadModel(path, optimize = false)

    override fun loadData(bytes: ByteArray, type: ONNXDataType): KIONNXData<*> {
        return when (type) {
            ONNXDataType.ONNX_TENSOR -> KITensor.create(protoReader(bytes).readTensor())
            ONNXDataType.ONNX_SEQUENCE -> KIONNXSequence.create(SequenceProto.decode(protoReader(bytes)))
            ONNXDataType.ONNX_MAP -> KIONNXMap.create(MapProto.decode(protoReader(bytes)))
        }
    }

    override suspend fun loadData(path: Path, type: ONNXDataType): KIONNXData<*> {
        return loadData(CommonDataLoader.bytes(path), type)
    }

    override suspend fun loadData(path: String, type: ONNXDataType): KIONNXData<*> {
        return loadData(path.toPath(), type)
    }
}
