package io.kinference.core

import io.kinference.BackendInfo
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.model.KIModel
import io.kinference.core.optimizer.rules.OptimizerRuleSet
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.IrOptimizableEngine
import io.kinference.optimizer.GraphOptimizer
import io.kinference.optimizer.OptimizerRule
import io.kinference.protobuf.*
import io.kinference.protobuf.message.*
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path
import okio.Path.Companion.toPath

typealias KIONNXData<T> = ONNXData<T, CoreBackend>

// Define an interface for allocation control marking output
internal interface KIONNXDataArraysReleaser {
    fun markContextOutput()
    fun markGlobalOutput()
}

internal fun <T> KIONNXData<T>.markContextOutput() {
    if (this is KIONNXDataArraysReleaser)
        this.markContextOutput()
}

internal fun <T> KIONNXData<T>.markGlobalOutput() {
    if (this is KIONNXDataArraysReleaser)
        this.markGlobalOutput()
}

object CoreBackend : BackendInfo(name = "KInference Core CPU Backend")

/**
 * This is an inference engine for KInference Core backend implementation
 * which is most efficient to use with distilled models of a relatively small size.
 * KInference Core is a pure Kotlin implementation that requires anything but vanilla Kotlin to run and so is
 * an advisable option for JVM projects employing inference on users' machines due to the small number of dependencies and JVM-optimized computations.
 * Note that, despite the fact that KInference Core is available for JS projects,
 * it is highly recommended to use KInference TensorFlow.js backend instead for more performance.
 */
object KIEngine : IrOptimizableEngine<KIONNXData<*>> {
    override val info: BackendInfo = CoreBackend

    private val KI_READER_CONFIG = ProtobufReader.ReaderConfig(tensorDecoder = TiledTensorDecoder)

    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), KI_READER_CONFIG)

    override suspend fun loadModel(bytes: ByteArray, optimize: Boolean): KIModel {
        val rules = if (optimize) OptimizerRuleSet.DEFAULT_OPT_RULES else emptyList()
        return loadModel(bytes, rules)
    }

    override suspend fun loadModel(bytes: ByteArray, rules: List<OptimizerRule<KIONNXData<*>>>): KIModel {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        val model = KIModel(modelScheme)

        return if (rules.isNotEmpty()) {
            val newGraph = GraphOptimizer(model.graph).run(rules) as KIGraph
            KIModel(model.id, model.name, model.opSet, newGraph)
        } else {
            model
        }
    }

    override suspend fun loadModel(bytes: ByteArray): KIModel = loadModel(bytes, optimize = true)

    override suspend fun loadModel(path: Path, optimize: Boolean): KIModel {
        return loadModel(CommonDataLoader.bytes(path), optimize)
    }

    override suspend fun loadModel(path: Path): KIModel = loadModel(path, optimize = true)

    override suspend fun loadModel(path: Path, rules: List<OptimizerRule<KIONNXData<*>>>): KIModel {
        return loadModel(CommonDataLoader.bytes(path), rules)
    }

    override suspend fun loadModel(path: String, optimize: Boolean): KIModel {
        return loadModel(CommonDataLoader.bytes(path.toPath()), optimize)
    }

    override suspend fun loadModel(path: String): KIModel = loadModel(path, optimize = true)

    override suspend fun loadModel(path: String, rules: List<OptimizerRule<KIONNXData<*>>>): KIModel {
        return loadModel(CommonDataLoader.bytes(path.toPath()), rules)
    }

    override suspend fun loadData(bytes: ByteArray, type: ONNXDataType): KIONNXData<*> {
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
