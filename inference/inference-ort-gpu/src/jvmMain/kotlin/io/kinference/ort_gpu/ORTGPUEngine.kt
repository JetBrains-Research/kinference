package io.kinference.ort_gpu

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.kinference.BackendInfo
import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.ort_gpu.data.tensor.ORTGPUTensor
import io.kinference.ort_gpu.model.ORTGPUModel
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.ArrayFormat
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path
import okio.Path.Companion.toPath

typealias ORTGPUData<T> = ONNXData<T, ORTGPUBackend>

object ORTGPUBackend : BackendInfo("ONNXRuntime-GPU")

object ORTGPUEngine : InferenceEngine<ORTGPUData<*>> {
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions()

    override val info: BackendInfo = ORTGPUBackend

    private val ORTGPU_READER_CONFIG = ProtobufReader.ReaderConfig(tensorFormat = ArrayFormat.PRIMITIVE)

    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), ORTGPU_READER_CONFIG)

    init {
        options.addCUDA()
    }

    override fun loadModel(bytes: ByteArray, optimize: Boolean): ORTGPUModel {
        if (optimize)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        else
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
        val session = env.createSession(bytes)
        return ORTGPUModel(session)
    }

    override suspend fun loadModel(path: Path, optimize: Boolean): ORTGPUModel {
        if (optimize)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        else
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
        val session = env.createSession(path.toString(), options)
        return ORTGPUModel(session)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType)= when (type) {
        ONNXDataType.ONNX_TENSOR -> ORTGPUTensor.create(TensorProto.decode(protoReader(bytes)))
        else -> error("$type construction is not supported in OnnxRuntime Java API")
    }

    override suspend fun loadData(path: Path, type: ONNXDataType) = loadData(CommonDataLoader.bytes(path), type)
}
