package io.kinference.data.tensors


import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.types.ValueInfo
import okio.ByteString
import java.nio.ByteBuffer
import java.nio.ByteOrder

//TODO: support segments
//TODO: support external data
class Tensor(val data: NDArray, info: ValueInfo) : ONNXData(ONNXDataType.ONNX_TENSOR, info) {
    override fun rename(name: String): ONNXData {
        return Tensor(data, ValueInfo(info.typeInfo, name))
    }

    operator fun plus(other: Tensor): Tensor {
        require(this.data is NumberNDArray && other.data is NumberNDArray)
        return (this.data + other.data).asTensor()
    }

    operator fun minus(other: Tensor): Tensor {
        require(this.data is NumberNDArray && other.data is NumberNDArray)
        return (this.data - other.data).asTensor()
    }

    operator fun times(other: Tensor): Tensor {
        require(this.data is NumberNDArray && other.data is NumberNDArray)
        return (this.data * other.data).asTensor()
    }

    infix fun matmul(other: Tensor): Tensor {
        require(this.data is NumberNDArray && other.data is NumberNDArray)
        return (this.data matmul other.data).asTensor()
    }

    companion object {
        //TODO: complex, uint32/64 tensors
        @Suppress("UNCHECKED_CAST")
        fun create(proto: TensorProto, divider: Int = 1): Tensor {
            if (proto.dims.isNullOrEmpty()) return parseScalar(proto)

            val shape = proto.dims!!.toIntArray()
            val type = DataType.fromValue(proto.data_type ?: 0)
            val (array, size) = parseArrayWithSize(type, proto)

            return if (size == null || size == 0) {
//                require(proto.raw_data != null) { "Tensor without data" }
                val rawData = proto.raw_data ?: ByteString.EMPTY
                val buffer = ByteBuffer.wrap(rawData.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)

                when (type) {
                    DataType.DOUBLE -> {
                        val array = buffer.asDoubleBuffer()
                        DoubleNDArray(shape, divider) { array[it] }.asTensor(proto.name)
                    }
                    DataType.FLOAT -> {
                        val array = buffer.asFloatBuffer()
                        FloatNDArray(shape, divider) { array[it] }.asTensor(proto.name)
                    }
                    DataType.INT64 -> {
                        val array = buffer.asLongBuffer()
                        LongNDArray(shape, divider) { array[it] }.asTensor(proto.name)
                    }
                    DataType.INT32 -> {
                        val array = buffer.asIntBuffer()
                        IntNDArray(shape, divider) { array[it] }.asTensor(proto.name)
                    }
                    DataType.INT8 -> {
                        ByteNDArray(shape, divider) { buffer[it] }.asTensor(proto.name)
                    }
                    DataType.UINT8 -> {
                        UByteNDArray(shape, divider) { buffer[it].toUByte() }.asTensor(proto.name)
                    }
                    DataType.BOOL -> {
                        BooleanNDArray(shape, divider) { buffer[it] != 0.toByte() }.asTensor(proto.name)
                    }
                    DataType.STRING -> error("String data MUST not be present in raw_data field")
                    else -> error("Unsupported data type $type")
                }
            } else Tensor(array, type!!, shape, proto.name, divider)
        }

        internal operator fun invoke(value: Any?, type: DataType, dims: IntArray = IntArray(0), name: String? = "", divider: Int = 1): Tensor {
            val name = name ?: ""
            if (dims.isEmpty()) return createScalarNDArray(type.resolveLocalDataType(), value!!).asTensor(name)

            return when (type) {
                DataType.DOUBLE -> {
                    value as DoubleArray
                    DoubleNDArray(dims, divider) { value[it] }.asTensor(name)
                }
                DataType.FLOAT -> {
                    value as FloatArray
                    FloatNDArray(dims, divider) { value[it] }.asTensor(name)
                }
                DataType.INT32 -> {
                    value as IntArray
                    IntNDArray(dims, divider) { value[it] }.asTensor(name)
                }
                DataType.INT8 -> {
                    value as ByteArray
                    ByteNDArray(dims, divider) { value[it] }.asTensor(name)
                }
                DataType.UINT8 -> {
                    value as UByteArray
                    UByteNDArray(dims, divider) { value[it] }.asTensor(name)
                }
                DataType.INT64 -> {
                    value as LongArray
                    LongNDArray(dims, divider) { value[it] }.asTensor(name)
                }
                DataType.INT16 -> {
                    value as ShortArray
                    ShortNDArray(dims, divider) { value[it] }.asTensor(name)
                }
                DataType.BOOL -> {
                    value as BooleanArray
                    BooleanNDArray(dims, divider) { value[it] }.asTensor(name)
                }
                DataType.STRING -> {
                    value as List<String>
                    StringNDArray(dims) { value[it] }.asTensor(name)
                }
                else -> error("Unsupported data type $type")
            }
        }

        internal operator fun invoke(value: List<Any>, type: DataType): Tensor {
            val dims = intArrayOf(value.size)
            return Tensor(value, type, dims)
        }

        private fun parseArrayWithSize(type: DataType?, proto: TensorProto) = when (type) {
            DataType.DOUBLE -> proto.double_data to proto.double_data?.size
            DataType.FLOAT -> proto.float_data to proto.float_data?.size
            DataType.INT64 -> proto.int64_data to proto.int64_data?.size
            DataType.INT32 -> proto.int32_data to proto.int32_data?.size
            DataType.BOOL -> proto.int32_data?.toBooleanArray() to proto.int32_data?.size
            DataType.STRING -> proto.string_data to proto.string_data.size
            DataType.UINT8 -> proto.int32_data?.toUByteArray() to proto.int32_data?.size
            DataType.INT8 -> proto.int32_data?.toByteArray() to proto.int32_data?.size
            else -> error("Unsupported data type $type")
        }

        private fun parseScalar(proto: TensorProto): Tensor {
            val type = DataType.fromValue(proto.data_type ?: 0)
            val (array, size) = parseArrayWithSize(type, proto)
            return if (size == null || size == 0) {
                val buffer = proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN)
                when (type) {
                    DataType.DOUBLE -> DoubleNDArray.scalar(buffer.double)
                    DataType.FLOAT -> FloatNDArray.scalar(buffer.float)
                    DataType.INT64 -> LongNDArray.scalar(buffer.long)
                    DataType.INT32 -> IntNDArray.scalar(buffer.int)
                    DataType.INT16 -> ShortNDArray.scalar(buffer.short)
                    DataType.INT8 -> ByteNDArray.scalar(buffer.get())
                    DataType.UINT8 -> UByteNDArray.scalar(buffer.get().toUByte())
                    DataType.BOOL -> BooleanNDArray.scalar(buffer.get() != (0).toByte())
                    else -> error("Unsupported data type $type")
                }.asTensor(proto.name)
            } else when (type) {
                DataType.DOUBLE -> DoubleNDArray.scalar((array as DoubleArray)[0])
                DataType.FLOAT -> FloatNDArray.scalar((array as FloatArray)[0])
                DataType.INT64 -> LongNDArray.scalar((array as LongArray)[0])
                DataType.INT32 -> IntNDArray.scalar((array as IntArray)[0])
                DataType.INT16 -> ShortNDArray.scalar((array as ShortArray)[0])
                DataType.INT8 -> ByteNDArray.scalar((array as ByteArray)[0])
                DataType.UINT8 -> UByteNDArray.scalar((array as UByteArray)[0])
                DataType.BOOL -> BooleanNDArray.scalar((array as BooleanArray)[0])
                DataType.STRING -> StringNDArray.scalar((array as List<String>)[0])
                else -> error("Unsupported data type $type")
            }.asTensor(proto.name)
        }
    }
}
