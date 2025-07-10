package io.kinference.ort.data.tensor

import ai.onnxruntime.*
import io.kinference.data.ONNXDataType
import io.kinference.data.ONNXTensor
import io.kinference.ort.ORTBackend
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.toLongArray
import java.nio.*

class ORTTensor(name: String?, override val data: OnnxTensor) : ONNXTensor<OnnxTensor, ORTBackend>(name, data) {
    override val backend: ORTBackend = ORTBackend

    override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
    override fun rename(name: String): ORTTensor = ORTTensor(name, data)

    val shape: LongArray
        get() = data.info.shape

    override fun close() {
        data.close()
    }

    override suspend fun clone(newName: String?): ORTTensor {
        return when(data.info.type) {
            OnnxJavaType.DOUBLE -> invoke(toDoubleArray(), shape, newName)
            OnnxJavaType.FLOAT -> invoke(toFloatArray(), shape, newName)
            OnnxJavaType.INT32 -> invoke(toIntArray(), shape, newName)
            OnnxJavaType.INT8 -> invoke(toByteArray(), shape, newName)
            OnnxJavaType.INT64 -> invoke(toLongArray(), shape, newName)
            OnnxJavaType.INT16 -> invoke(toShortArray(), shape, newName)
            OnnxJavaType.UINT8 -> invoke(toUByteArray(), shape, newName)
            OnnxJavaType.BOOL -> invoke(toBooleanArray(), shape, newName)
            else -> error("Unsupported data type: $type")
        }
    }

    fun toDoubleArray(): DoubleArray {
        require(data.info.type == OnnxJavaType.DOUBLE) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        return data.doubleBuffer.array()
    }

    fun toFloatArray(): FloatArray {
        require(data.info.type == OnnxJavaType.FLOAT) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        return data.floatBuffer.array()
    }

    fun toIntArray(): IntArray {
        require(data.info.type == OnnxJavaType.INT32) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        return data.intBuffer.array()
    }

    fun toByteArray(): ByteArray {
        require(data.info.type == OnnxJavaType.INT8) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        return data.byteBuffer.array()
    }

    fun toLongArray(): LongArray {
        require(data.info.type == OnnxJavaType.INT64) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        return data.longBuffer.array()
    }

    fun toShortArray(): ShortArray {
        require(data.info.type == OnnxJavaType.INT16) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        return data.shortBuffer.array()
    }

    /*fun toStringArray(): String {
        require(data.info.type == OnnxJavaType.STRING) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        return data.value as String
    }*/

    fun toUByteArray(): UByteArray {
        require(data.info.type == OnnxJavaType.UINT8) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        val byteBuffer = data.byteBuffer
        return UByteArray(byteBuffer.capacity()) { byteBuffer.get().toUByte() }
    }

    fun toBooleanArray(): BooleanArray {
        require(data.info.type == OnnxJavaType.BOOL) { "Incompatible tensor type. Current tensor type: ${data.info.type}" }
        val byteBuffer = data.byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        return BooleanArray(byteBuffer.capacity()) { byteBuffer.get() == BYTE_ONE }
    }

    companion object {
        suspend fun create(proto: TensorProto): ORTTensor {
            val type = proto.dataType ?: TensorProto.DataType.UNDEFINED
            val array = if (proto.isString()) proto.stringData else proto.getArrayData()

            requireNotNull(array) { "Array value should be initialized" }

            return ORTTensor(array, type, proto.dims.toLongArray(), proto.name)
        }

        operator fun invoke(array: DoubleArray, dims: LongArray, name: String? = null): ORTTensor {
            val buffer = DoubleBuffer.wrap(array)
            val onnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
            return ORTTensor(name, onnxTensor)
        }

        operator fun invoke(array: FloatArray, dims: LongArray, name: String? = null): ORTTensor {
            val buffer = FloatBuffer.wrap(array)
            val onnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
            return ORTTensor(name, onnxTensor)
        }

        operator fun invoke(array: IntArray, dims: LongArray, name: String? = null): ORTTensor {
            val buffer = IntBuffer.wrap(array)
            val onnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
            return ORTTensor(name, onnxTensor)
        }

        operator fun invoke(array: ByteArray, dims: LongArray, name: String? = null): ORTTensor {
            val buffer = ByteBuffer.wrap(array)
            val onnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
            return ORTTensor(name, onnxTensor)
        }

        operator fun invoke(array: LongArray, dims: LongArray, name: String? = null): ORTTensor {
            val buffer = LongBuffer.wrap(array)
            val onnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
            return ORTTensor(name, onnxTensor)
        }

        operator fun invoke(array: ShortArray, dims: LongArray, name: String? = null): ORTTensor {
            val buffer = ShortBuffer.wrap(array)
            val onnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
            return ORTTensor(name, onnxTensor)
        }

        operator fun invoke(array: UByteArray, dims: LongArray, name: String? = null): ORTTensor {
            val buffer = ByteBuffer.allocateDirect(array.size).apply {
                for (number in array) put(number.toByte())
            }
            val onnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims, OnnxJavaType.UINT8)
            return ORTTensor(name, onnxTensor)
        }

        private const val BYTE_ONE = (1).toByte()
        private const val BYTE_ZERO = (0).toByte()

        operator fun invoke(array: BooleanArray, dims: LongArray, name: String? = null): ORTTensor {
            val buffer = ByteBuffer.allocateDirect(array.size).order(ByteOrder.LITTLE_ENDIAN)
            for (bool in array) buffer.put(if (bool) BYTE_ONE else BYTE_ZERO)
            buffer.rewind()
            val onnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims, OnnxJavaType.BOOL)
            return ORTTensor(name, onnxTensor)
        }

        private operator fun invoke(value: Any, type: TensorProto.DataType, dims: LongArray = LongArray(0), name: String? = null): ORTTensor =
            when(type) {
                TensorProto.DataType.DOUBLE -> invoke(value as DoubleArray, dims, name)
                TensorProto.DataType.FLOAT -> invoke(value as FloatArray, dims, name)
                TensorProto.DataType.INT32 -> invoke(value as IntArray, dims, name)
                TensorProto.DataType.INT8 -> invoke(value as ByteArray, dims, name)
                TensorProto.DataType.INT64 -> invoke(value as LongArray, dims, name)
                TensorProto.DataType.INT16 -> invoke(value as ShortArray, dims, name)
//                TensorProto.DataType.STRING -> invoke(value as List<String>, dims, name)
                TensorProto.DataType.UINT8 -> invoke(value as UByteArray, dims, name)
                TensorProto.DataType.BOOL -> invoke(value as BooleanArray, dims, name)
                else -> error("Unsupported data type $type")
            }
    }
}
