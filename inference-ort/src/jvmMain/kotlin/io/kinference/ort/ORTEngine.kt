package io.kinference.ort

import ai.onnxruntime.*
import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ort.data.ORTTensor
import io.kinference.ort.model.ORTModel
import io.kinference.protobuf.message.TensorProto
import java.nio.*

//TODO: Support session options
object ORTEngine : InferenceEngine {
    override fun loadModel(bytes: ByteArray): Model {
        val session = OrtEnvironment.getEnvironment().createSession(bytes)
        return ORTModel(session)
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): ONNXData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> loadTensor(bytes)
        else -> error("$type construction is not supported in OnnxRuntime Java API")
    }

    private fun loadTensor(bytes: ByteArray): ORTTensor {
        val proto = TensorProto.decode(bytes)
        val type = proto.dataType ?: TensorProto.DataType.UNDEFINED
        val array = when {
            proto.isTiled() -> proto.tiledData
            proto.isString() -> proto.stringData
            else -> error("Unsupported data type ${proto.dataType}")
        }

        requireNotNull(array) { "Array value should be initialized" }

        return readTensor(array, type, proto.dims, proto.name)
    }

    private fun IntArray.toLongArray() = LongArray(this.size) { this[it].toLong() }

    private fun readTensor(value: Any, type: TensorProto.DataType, dims: IntArray = IntArray(0), name: String? = ""): ORTTensor {
        val buffer = when (type) {
            TensorProto.DataType.DOUBLE -> {
                val buffer = DoubleBuffer.wrap((value as DoubleTiledArray).toArray())
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims.toLongArray())
            }
            TensorProto.DataType.FLOAT -> {
                val buffer = FloatBuffer.wrap((value as FloatTiledArray).toArray())
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims.toLongArray())
            }
            TensorProto.DataType.INT32 -> {
                val buffer = IntBuffer.wrap((value as IntTiledArray).toArray())
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims.toLongArray())
            }
            TensorProto.DataType.INT8 -> {
                val buffer = ByteBuffer.wrap((value as ByteTiledArray).toArray())
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims.toLongArray())
            }
            TensorProto.DataType.INT64 -> {
                val buffer = LongBuffer.wrap((value as LongTiledArray).toArray())
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims.toLongArray())
            }
            TensorProto.DataType.INT16 -> {
                val buffer = ShortBuffer.wrap((value as ShortTiledArray).toArray())
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims.toLongArray())
            }
            TensorProto.DataType.STRING -> {
                val array = (value as List<String>).toTypedArray()
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), array, dims.toLongArray())
            }
            else -> error("Unsupported data type $type")
        }
        return ORTTensor(buffer, name ?: "")
    }
}
