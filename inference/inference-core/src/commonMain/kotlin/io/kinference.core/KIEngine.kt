package io.kinference.core

import io.kinference.*
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.model.KIModel
import io.kinference.core.optimizer.rules.DequantizeMatMulInteger
import io.kinference.core.optimizer.rules.DequantizeQAttention
import io.kinference.data.*
import io.kinference.model.Model
import io.kinference.optimizer.*
import io.kinference.protobuf.*
import io.kinference.protobuf.message.*
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path
import kotlin.time.ExperimentalTime

typealias KIONNXData<T> = ONNXData<T, CoreBackend>

object CoreBackend : BackendInfo(name = "KInference Core CPU Backend")

@OptIn(ExperimentalTime::class)
object KIEngine : OptimizableEngine<KIONNXData<*>> {
    override val info: BackendInfo = CoreBackend

    private val KI_READER_CONFIG = ProtobufReader.ReaderConfig(tensorDecoder = TiledTensorDecoder)
    private val defaultOptRules = listOf(DequantizeMatMulInteger, DequantizeQAttention)
    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), KI_READER_CONFIG)

    override fun loadModel(bytes: ByteArray): KIModel {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return KIModel(modelScheme)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): KIONNXData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> KITensor.create(protoReader(bytes).readTensor())
        ONNXDataType.ONNX_SEQUENCE -> KIONNXSequence.create(SequenceProto.decode(protoReader(bytes)))
        ONNXDataType.ONNX_MAP -> KIONNXMap.create(MapProto.decode(protoReader(bytes)))
    }

    private fun parseOptLevel(level: OptLevel): List<OptimizerRule<KIONNXData<*>>> = when (level) {
        OptLevel.NO_OPT -> emptyList()
        OptLevel.DEFAULT, OptLevel.ALL -> defaultOptRules
    }

    override fun optimizeModel(model: Model<KIONNXData<*>>, level: OptLevel): Model<KIONNXData<*>> {
        val rules = parseOptLevel(level)
        return optimizeModel(model, rules)
    }

    override fun optimizeModel(model: Model<KIONNXData<*>>, rules: List<OptimizerRule<KIONNXData<*>>>): Model<KIONNXData<*>> {
        val newGraph = GraphOptimizer((model as KIModel).graph).run(rules) as KIGraph
        return KIModel(model.name, model.opSet, newGraph)
    }

    override suspend fun loadData(path: Path, type: ONNXDataType) = loadData(CommonDataLoader.bytes(path), type)
    override suspend fun loadModel(path: Path) = loadModel(CommonDataLoader.bytes(path))
}
