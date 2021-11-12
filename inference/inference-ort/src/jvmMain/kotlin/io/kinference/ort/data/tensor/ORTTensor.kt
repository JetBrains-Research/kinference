package io.kinference.ort.data.tensor

import ai.onnxruntime.*
import io.kinference.data.*
import io.kinference.ndarray.extensions.primitiveFromTiledArray
import io.kinference.ort.ORTBackend
import io.kinference.protobuf.message.TensorProto
import java.nio.*

class ORTTensor(name: String?, override val data: OnnxTensor) : ONNXTensor<OnnxTensor, ORTBackend>(name, data) {
    override val backend: ORTBackend = ORTBackend

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
                TensorProto.DataType.UINT8 -> {
                    value as UByteArray
                    val buffer = ByteBuffer.allocate(value.size).apply {
                        for (number in value) put(number.toByte())
                    }
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims, OnnxJavaType.UINT8)
                }
                TensorProto.DataType.BOOL -> {
                    value as BooleanArray
                    val buffer = ByteBuffer.allocateDirect(OnnxJavaType.BOOL.size * value.size).order(ByteOrder.LITTLE_ENDIAN)
                    for (b in value) buffer.put(if (b) (1).toByte() else (0).toByte())
                    buffer.rewind()
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims, OnnxJavaType.BOOL)
                }
                else -> error("Unsupported data type $type")
            }
            return ORTTensor(name, onnxTensor)
        }
    }
}
