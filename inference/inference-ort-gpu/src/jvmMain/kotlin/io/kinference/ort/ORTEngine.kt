package io.kinference.ort

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession.SessionOptions
import io.kinference.BackendInfo
import io.kinference.OptimizableEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.ort.model.ORTModel
import io.kinference.protobuf.*
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path
import okio.Path.Companion.toPath

typealias ORTData<T> = ONNXData<T, ORTBackend>

object ORTBackend : BackendInfo("ONNXRuntime-GPU")

/**
 * This is an inference engine for KInference Java ONNXRuntime GPU backend implementation.
 * This backend provides common KInference API to interact with ONNXRuntime library.
 *
 * Note that the backend is **CUDA-only** and uses JNI for model inference.
 * To check on the system requirements, visit the following [link](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
 *
 * ONNXRuntime version: 1.13.1
 */
object ORTEngine : OptimizableEngine<ORTData<*>> {
    private val env = OrtEnvironment.getEnvironment()
    private val options = SessionOptions()

    override val info: BackendInfo = ORTBackend

    private val ORTGPU_READER_CONFIG = ProtobufReader.ReaderConfig(tensorDecoder = FlatTensorDecoder)

    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), ORTGPU_READER_CONFIG)

    init {
        options.addCUDA()
    }

    override suspend fun loadModel(bytes: ByteArray): ORTModel {
        val session = env.createSession(bytes, options)
        return ORTModel(session)
    }

    override suspend fun loadModel(bytes: ByteArray, optimize: Boolean): ORTModel {
        if (optimize)
            options.setOptimizationLevel(SessionOptions.OptLevel.BASIC_OPT)
        else
            options.setOptimizationLevel(SessionOptions.OptLevel.NO_OPT)
        val session = env.createSession(bytes, options)
        return ORTModel(session)
    }

    fun loadModel(bytes: ByteArray, optimize: Boolean, logLevel: OrtLoggingLevel =  OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO): ORTModel {
        if (optimize)
            options.setOptimizationLevel(SessionOptions.OptLevel.BASIC_OPT)
        else
            options.setOptimizationLevel(SessionOptions.OptLevel.NO_OPT)

        options.setSessionLogLevel(logLevel)
        val session = env.createSession(bytes, options)

        return ORTModel(session)
    }

    fun loadModel(bytes: ByteArray, options: SessionOptions): ORTModel {
        val session = env.createSession(bytes, options)

        return ORTModel(session)
    }

    override suspend fun loadModel(path: Path, optimize: Boolean): ORTModel {
        if (optimize)
            options.setOptimizationLevel(SessionOptions.OptLevel.BASIC_OPT)
        else
            options.setOptimizationLevel(SessionOptions.OptLevel.NO_OPT)
        val session = env.createSession(path.toString(), options)
        return ORTModel(session)
    }

    override suspend fun loadModel(path: Path): ORTModel {
        val session = env.createSession(path.toString(), options)
        return ORTModel(session)
    }

    override suspend fun loadModel(path: String): ORTModel {
        return loadModel(path.toPath())
    }

    override suspend fun loadModel(path: String, optimize: Boolean): ORTModel {
        return loadModel(path.toPath(), optimize)
    }

    fun loadModel(path: Path, options: SessionOptions): ORTModel {
        val session = env.createSession(path.toString(), options)
        return ORTModel(session)
    }

    fun loadModel(path: String, options: SessionOptions): ORTModel {
        return loadModel(path.toPath(), options)
    }

    override suspend fun loadData(bytes: ByteArray, type: ONNXDataType): ORTData<*> {
        return when (type) {
            ONNXDataType.ONNX_TENSOR -> ORTTensor.create(protoReader(bytes).readTensor())
            else -> error("$type construction is not supported in OnnxRuntime Java API")
        }
    }

    override suspend fun loadData(path: Path, type: ONNXDataType): ORTData<*> {
        return loadData(CommonDataLoader.bytes(path), type)
    }

    override suspend fun loadData(path: String, type: ONNXDataType): ORTData<*> {
        return loadData(path.toPath(), type)
    }
}
