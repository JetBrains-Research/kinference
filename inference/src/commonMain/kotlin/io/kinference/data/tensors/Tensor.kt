package io.kinference.data.tensors


import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.toIntArray
import io.kinference.onnx.TensorProto
import io.kinference.onnx.TensorProto.DataType
import io.kinference.utils.readDoubleLe
import io.kinference.utils.readFloatLe
import io.kinference.types.ValueInfo
import okio.*

//import java.nio.ByteBuffer
//import java.nio.ByteOrder

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

            val shape = proto.dims.toIntArray()
            val type = DataType.fromValue(proto.data_type ?: 0)
            val array = when (type) {
                DataType.DOUBLE -> proto.double_data
                DataType.FLOAT -> proto.float_data
                DataType.INT64 -> proto.int64_data
                DataType.INT32 -> proto.int32_data
                DataType.BOOL -> proto.int32_data.map { it != 0 }
                DataType.STRING -> proto.string_data.map { it.utf8() }
                DataType.UINT8 -> proto.int32_data.map { it.toUByte() }
                DataType.INT8 -> proto.int32_data.map { it.toByte() }
                else -> error("Unsupported data type $type")
            }

            return if (array.isEmpty()) {
//                require(proto.raw_data != null) { "Tensor without data" }
                val rawData = proto.raw_data ?: ByteString.EMPTY
                val buffer = Buffer().apply { write(rawData) }

                when (type) {
                    DataType.DOUBLE -> {
                        DoubleNDArray(shape, divider) { buffer.readDoubleLe() }.asTensor(proto.name)
                    }
                    DataType.FLOAT -> {
                        FloatNDArray(shape, divider) { buffer.readFloatLe() }.asTensor(proto.name)
                    }
                    DataType.INT64 -> {
                        LongNDArray(shape, divider) { buffer.readLongLe() }.asTensor(proto.name)
                    }
                    DataType.INT32 -> {
                        IntNDArray(shape, divider) { buffer.readIntLe() }.asTensor(proto.name)
                    }
                    DataType.INT8 -> {
                        ByteNDArray(shape, divider) { buffer.readByte() }.asTensor(proto.name)
                    }
                    DataType.UINT8 -> {
                        UByteNDArray(shape, divider) { buffer.readByte().toUByte() }.asTensor(proto.name)
                    }
                    DataType.BOOL -> {
                        BooleanNDArray(shape, divider) { buffer.readByte() != 0.toByte() }.asTensor(proto.name)
                    }
                    DataType.STRING -> error("String data MUST not be present in raw_data field")
                    else -> error("Unsupported data type $type")
                }
            } else when (type) {
                DataType.DOUBLE -> Tensor(array as List<Double>, type, shape, proto.name, divider)
                DataType.FLOAT -> Tensor(array as List<Float>, type, shape, proto.name, divider)
                DataType.INT64 -> Tensor(array as List<Long>, type, shape, proto.name, divider)
                DataType.INT32 -> Tensor(array as List<Int>, type, shape, proto.name, divider)
                DataType.INT8 -> Tensor(array as List<Byte>, type, shape, proto.name, divider)
                DataType.UINT8 -> Tensor(array as List<UByte>, type, shape, proto.name, divider)
                DataType.BOOL -> Tensor(array as List<Boolean>, type, shape, proto.name, divider)
                DataType.STRING -> Tensor(array as List<String>, type, shape, proto.name, divider)
                else -> error("Unsupported data type $type")
            }
        }

        internal operator fun invoke(dims: List<Long>, value: List<Any?>, type: DataType, name: String?, divider: Int = 1): Tensor {
            val shape = dims.toIntArray()

            return if (dims.isEmpty()) {
                createScalarNDArray(type.resolveLocalDataType(), value[0]!!)
            } else {
                val data = createArray(type.resolveLocalDataType(), shape, divider) { i -> value[i]!! }
                createNDArray(type.resolveLocalDataType(), data, shape)
            }.asTensor(name)
        }

        internal operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0), name: String? = "", divider: Int = 1): Tensor {
            val name = name ?: ""
            if (dims.isEmpty()) return createScalarNDArray(type.resolveLocalDataType(), value).asTensor(name)

            value as List<Any?>
            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(dims, divider) { value[it] as Double }.asTensor(name)
                DataType.FLOAT -> FloatNDArray(dims, divider) { value[it] as Float }.asTensor(name)
                DataType.INT32 -> IntNDArray(dims, divider) { value[it] as Int }.asTensor(name)
                DataType.INT8 -> ByteNDArray(dims, divider) { value[it] as Byte }.asTensor(name)
                DataType.UINT8 -> UByteNDArray(dims, divider) { value[it] as UByte }.asTensor(name)
                DataType.INT64 -> LongNDArray(dims, divider) { value[it] as Long }.asTensor(name)
                DataType.INT16 -> ShortNDArray(dims, divider) { value[it] as Short }.asTensor(name)
                DataType.BOOL -> BooleanNDArray(dims, divider) { value[it] as Boolean }.asTensor(name)
                else -> error("Unsupported data type $type")
            }
        }

        internal operator fun invoke(value: List<Any>, type: DataType): Tensor {
            val dims = intArrayOf(value.size)
            return Tensor(value, type, dims)
        }

        private fun parseScalar(proto: TensorProto): Tensor {
            val type = DataType.fromValue(proto.data_type ?: 0)
            val array = when (type) {
                DataType.DOUBLE -> proto.double_data
                DataType.FLOAT -> proto.float_data
                DataType.INT64 -> proto.int64_data
                DataType.INT32 -> proto.int32_data
                DataType.INT8 -> if (!proto.int32_data.isNullOrEmpty()) proto.int32_data.map { it.toByte() } else emptyList()
                DataType.UINT8 -> if (!proto.int32_data.isNullOrEmpty()) proto.int32_data.map { it.toUByte() } else emptyList()
                DataType.BOOL -> proto.int32_data.map { it != 0 }
                else -> error("Unsupported data type $type")
            }
            return if (array.isEmpty()) {
                val buffer = Buffer().apply { write(proto.raw_data!!) }
                when (type) {
                    DataType.DOUBLE -> DoubleNDArray.scalar(buffer.readDoubleLe())
                    DataType.FLOAT -> FloatNDArray.scalar(buffer.readFloatLe())
                    DataType.INT64 -> LongNDArray.scalar(buffer.readLongLe())
                    DataType.INT32 -> IntNDArray.scalar(buffer.readIntLe())
                    DataType.INT16 -> ShortNDArray.scalar(buffer.readShortLe())
                    DataType.INT8 -> ByteNDArray.scalar(buffer.readByte())
                    DataType.UINT8 -> UByteNDArray.scalar(buffer.readByte().toUByte())
                    DataType.BOOL -> BooleanNDArray.scalar(buffer.readByte() != (0).toByte())
                    else -> error("Unsupported data type $type")
                }.asTensor(proto.name)
            } else createScalarNDArray(type.resolveLocalDataType(), array[0]).asTensor(proto.name)
        }
    }
}
