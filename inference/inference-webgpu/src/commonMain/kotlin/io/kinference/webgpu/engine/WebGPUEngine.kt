package io.kinference.webgpu.engine

import io.kinference.BackendInfo
import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.ArrayFormat
import io.kinference.protobuf.message.ModelProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.webgpu.Device
import io.kinference.webgpu.tensor.WebGPUTensor
import io.kinference.webgpu.model.WebGPUModel
import okio.Buffer
import kotlin.time.ExperimentalTime

typealias WebGPUData<T> = ONNXData<T, WebGPUBackend>

object WebGPUBackend : BackendInfo(name = "WebGPU")

@OptIn(ExperimentalTime::class)
object WebGPUEngine : InferenceEngine<WebGPUData<*>> {
    override val info = WebGPUBackend

    private val WEBGPU_READER_CONFIG = ProtobufReader.ReaderConfig(tensorFormat = ArrayFormat.PRIMITIVE)
    private fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), WEBGPU_READER_CONFIG)

    override suspend fun loadDataSuspend(bytes: ByteArray, type: ONNXDataType): WebGPUData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> WebGPUTensor.create(TensorProto.decode(protoReader(bytes)), WebGPUEnvironment.getDevice())
        else -> TODO()
    }

    override suspend fun loadModelSuspend(bytes: ByteArray): Model<WebGPUData<*>> {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return WebGPUModel(modelScheme, WebGPUEnvironment.getDevice())
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*, *> {
        error("Use loadDataSuspend with WebGPU backend")
    }

    override fun loadModel(bytes: ByteArray): Model<WebGPUData<*>> {
        error("Use loadModelSuspend with WebGPU backend")
    }
}
