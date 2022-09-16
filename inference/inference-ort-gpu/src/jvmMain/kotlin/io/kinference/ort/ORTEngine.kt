package io.kinference.ort

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.kinference.BackendInfo
import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.ort.model.ORTModel
import io.kinference.protobuf.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path

typealias ORTData<T> = ONNXData<T, ORTBackend>

object ORTBackend : BackendInfo("ONNXRuntime-GPU")

object ORTEngine : InferenceEngine<ORTData<*>> {
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions()

    override val info: BackendInfo = ORTBackend

    private val ORTGPU_READER_CONFIG = ProtobufReader.ReaderConfig(tensorDecoder = FlatTensorDecoder)

    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), ORTGPU_READER_CONFIG)

    init {
        options.addCUDA()
    }

    override fun loadModel(bytes: ByteArray): Model<ORTData<*>> {
        val session = env.createSession(bytes)
        return ORTModel(session)
    }

    fun loadModel(bytes: ByteArray, optimize: Boolean): ORTModel {
        if (optimize)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        else
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
        val session = env.createSession(bytes)
        return ORTModel(session)
    }

    fun loadModel(path: Path, optimize: Boolean): ORTModel {
        if (optimize)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        else
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
        val session = env.createSession(path.toString(), options)
        return ORTModel(session)
    }

    override suspend fun loadModel(path: Path): Model<ORTData<*>> {
        val session = env.createSession(path.toString(), options)
        return ORTModel(session)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType)= when (type) {
        ONNXDataType.ONNX_TENSOR -> ORTTensor.create(protoReader(bytes).readTensor())
        else -> error("$type construction is not supported in OnnxRuntime Java API")
    }

    override suspend fun loadData(path: Path, type: ONNXDataType) = loadData(CommonDataLoader.bytes(path), type)
}
