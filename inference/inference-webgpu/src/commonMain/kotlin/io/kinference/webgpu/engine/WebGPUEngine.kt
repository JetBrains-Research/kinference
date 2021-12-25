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
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.model.WebGPUModel
import okio.Buffer

typealias WebGPUData<T> = ONNXData<T, WebGPUBackend>

object WebGPUBackend : BackendInfo(name = "WebGPU")

object WebGPUEngine : InferenceEngine<WebGPUData<*>> {
    override val info = WebGPUBackend

    private val WEBGPU_READER_CONFIG = ProtobufReader.ReaderConfig(tensorFormat = ArrayFormat.PRIMITIVE)
    private fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), WEBGPU_READER_CONFIG)

    override fun loadData(bytes: ByteArray, type: ONNXDataType): WebGPUData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> WebGPUTensor.create(TensorProto.decode(protoReader(bytes)))
        else -> TODO()
    }

    override fun loadModel(bytes: ByteArray): Model<WebGPUData<*>> {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return WebGPUModel(modelScheme)
    }
}
