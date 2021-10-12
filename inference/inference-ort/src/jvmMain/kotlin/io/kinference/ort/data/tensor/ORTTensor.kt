package io.kinference.ort.data.tensor

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import io.kinference.data.ONNXDataType
import io.kinference.data.ONNXTensor
import io.kinference.ndarray.extensions.primitiveFromTiledArray
import io.kinference.protobuf.message.TensorProto
import java.nio.*

class ORTTensor(name: String?, override val data: OnnxTensor) : ONNXTensor<OnnxTensor>(name, data) {
    override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
    override fun rename(name: String): ORTTensor = ORTTensor(name, data)

    companion object {
        fun create(proto: TensorProto): ORTTensor {
            val type = proto.dataType ?: TensorProto.DataType.UNDEFINED
            val array = when {
                proto.isTiled() -> primitiveFromTiledArray(proto.arrayData!!)
                proto.isString() -> proto.stringData
                proto.isPrimitive() -> proto.arrayData
                else -> error("Unsupported data type ${proto.dataType}")
            }
            requireNotNull(array) { "Array value should be initialized" }

            return ORTTensor(array, type, proto.dims.toLongArray(), proto.name)
        }

        private fun IntArray.toLongArray() = LongArray(this.size) { this[it].toLong() }

        private operator fun invoke(value: Any, type: TensorProto.DataType, dims: LongArray = LongArray(0), name: String? = null): ORTTensor {
            val onnxTensor = when (type) {
                TensorProto.DataType.DOUBLE -> {
                    val buffer = DoubleBuffer.wrap(value as DoubleArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.FLOAT -> {
                    val buffer = FloatBuffer.wrap(value as FloatArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.INT32 -> {
                    val buffer = IntBuffer.wrap(value as IntArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.INT8 -> {
                    val buffer = ByteBuffer.wrap(value as ByteArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.INT64 -> {
                    val buffer = LongBuffer.wrap(value as LongArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.INT16 -> {
                    val buffer = ShortBuffer.wrap(value as ShortArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.STRING -> {
                    val array = (value as List<String>).toTypedArray()
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), array, dims)
                }
                else -> error("Unsupported data type $type")
            }
            return ORTTensor(name ?: "", onnxTensor)
        }
    }
}
