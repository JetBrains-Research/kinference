package io.kinference.ort

import ai.onnxruntime.*
import io.kinference.BackendInfo
import io.kinference.InferenceEngine
import io.kinference.data.*
import io.kinference.model.Model
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.ort.model.ORTModel
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.ArrayFormat
import io.kinference.protobuf.message.TensorProto
import okio.Buffer

typealias ORTData<T> = ONNXData<T, ORTBackend>

object ORTBackend : BackendInfo(name = "ONNXRuntime")

//TODO: Support session options
object ORTEngine : InferenceEngine<ORTData<*>> {
    override val info: BackendInfo = ORTBackend

    private val ORT_READER_CONFIG = ProtobufReader.ReaderConfig(tensorFormat = ArrayFormat.PRIMITIVE)
    fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), ORT_READER_CONFIG)

    override fun loadModel(bytes: ByteArray): Model<ORTData<*>> {
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        val session = env.createSession(bytes, options)
        return ORTModel(session)
    }

    fun loadModel(bytes: ByteArray, options: OrtSession.SessionOptions): Model<ORTData<*>> {
        val session = OrtEnvironment.getEnvironment().createSession(bytes, options)
        return ORTModel(session)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): ORTData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> ORTTensor.create(TensorProto.decode(protoReader(bytes)))
        else -> error("$type construction is not supported in OnnxRuntime Java API")
    }
}