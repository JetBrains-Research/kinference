package io.kinference.data.tensors

import io.kinference.ndarray.NDArray
import io.kinference.ndarray.NumberNDArray
import io.kinference.ndarray.Strides
import io.kinference.ndarray.toIntArray
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.extensions.createArray
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.ndarray.extensions.createScalarNDArray
import io.kinference.ndarray.extensions.matmul
import io.kinference.ndarray.extensions.*
import io.kinference.onnx.TensorProto
import io.kinference.onnx.TensorProto.DataType
import io.kinference.types.TensorInfo
import io.kinference.types.TensorShape
import java.nio.ByteBuffer
import java.nio.ByteOrder

//TODO: support segments
//TODO: support external and raw data
class Tensor(val data: NDArray, info: TensorInfo) : ONNXData(ONNXDataType.ONNX_TENSOR, info) {
    override fun rename(newName: String): ONNXData {
        return Tensor(data, TensorInfo(newName, info.type, TensorShape(data.shape)))
    }

    @ExperimentalUnsignedTypes
    fun mapElements(type: DataType = this.info.type, func: (Any) -> Any): Tensor {
        val localType = type.resolveLocalDataType()
        val buffer = createArray(localType, data.linearSize) { func(data[it]) }
        return createNDArray(localType, buffer, data.shape).asTensor()
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
        @ExperimentalUnsignedTypes
        fun create(proto: TensorProto): Tensor {
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
                DataType.UINT8 -> emptyList<UByte>()
                DataType.INT8 -> emptyList<Byte>()
                else -> error("Unsupported data type $type")
            }

            return if (array.isEmpty()) {
                require(proto.raw_data != null) { "Tensor without data" }
                val buffer = ByteBuffer.wrap(proto.raw_data.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
                val sizeInBytes = proto.raw_data.size

                when (type) {
                    DataType.DOUBLE -> {
                        val array = DoubleArray(sizeInBytes / 8)
                        buffer.asDoubleBuffer().get(array)
                        Tensor(array, type, shape, proto.name)
                    }
                    DataType.FLOAT -> {
                        val array = FloatArray(sizeInBytes / 4)
                        buffer.asFloatBuffer().get(array)
                        Tensor(array, type, shape, proto.name)
                    }
                    DataType.INT64 -> {
                        val array = LongArray(sizeInBytes / 8)
                        buffer.asLongBuffer().get(array)
                        Tensor(array, type, shape, proto.name)
                    }
                    DataType.INT32 -> {
                        val array = IntArray(sizeInBytes / 4)
                        buffer.asIntBuffer().get(array)
                        Tensor(array, type, shape, proto.name)
                    }
                    DataType.INT8 -> {
                        val array = ByteArray(sizeInBytes)
                        buffer.get(array)
                        Tensor(array, type, shape, proto.name)
                    }
                    DataType.UINT8 -> {
                        val array = ByteArray(sizeInBytes)
                        buffer.get(array)
                        Tensor(array.toUByteArray(), type, shape, proto.name)
                    }
                    DataType.BOOL -> Tensor(BooleanArray(sizeInBytes) { buffer[it] != 0.toByte() }, type, shape, proto.name)
                    DataType.STRING -> error("String data MUST not be present in raw_data field")
                    else -> error("Unsupported data type $type")
                }
            } else when (type) {
                DataType.DOUBLE -> Tensor((array as List<Double>).toDoubleArray(), type, shape, proto.name)
                DataType.FLOAT -> Tensor((array as List<Float>).toFloatArray(), type, shape, proto.name)
                DataType.INT64 -> Tensor((array as List<Long>).toLongArray(), type, shape, proto.name)
                DataType.INT32 -> Tensor((array as List<Int>).toIntArray(), type, shape, proto.name)
                DataType.BOOL -> Tensor((array as List<Boolean>).toBooleanArray(), type, shape, proto.name)
                DataType.STRING -> Tensor((array as List<String>).toTypedArray(), type, shape, proto.name)
                else -> error("Unsupported data type $type")
            }
        }

        @ExperimentalUnsignedTypes
        operator fun invoke(dims: List<Long>, value: List<*>, type: DataType, name: String?): Tensor {
            val data = createArray(type.resolveLocalDataType(), value.size) { i -> value[i]!! }
            return Tensor(data, type, dims.toIntArray(), name)
        }


        @ExperimentalUnsignedTypes
        operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0), name: String? = ""): Tensor {
            val name = name ?: ""
            if (dims.isEmpty()) return createScalarNDArray(type.resolveLocalDataType(), value).asTensor(name)

            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(value as DoubleArray, Strides(dims)).asTensor(name)
                DataType.FLOAT -> FloatNDArray(value as FloatArray, Strides(dims)).asTensor(name)
                DataType.INT32 -> IntNDArray(value as IntArray, Strides(dims)).asTensor(name)
                DataType.INT8 -> ByteNDArray(value as ByteArray, Strides(dims)).asTensor(name)
                DataType.UINT8 -> UByteNDArray(value as UByteArray, Strides(dims)).asTensor(name)
                DataType.INT64 -> LongNDArray(value as LongArray, Strides(dims)).asTensor(name)
                DataType.INT16 -> ShortNDArray(value as ShortArray, Strides(dims)).asTensor(name)
                DataType.BOOL -> BooleanNDArray(value as BooleanArray, Strides(dims)).asTensor(name)
                else -> error("Unsupported data type $type")
            }
        }

        @ExperimentalUnsignedTypes
        operator fun invoke(value: List<Any>, type: DataType): Tensor {
            val dims = intArrayOf(value.size)
            val data = createArray(type.resolveLocalDataType(), value.size) { i -> value[i] }
            return Tensor(data, type, dims)
        }

        @ExperimentalUnsignedTypes
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
                when (type) {
                    DataType.DOUBLE -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).double, type, dims = IntArray(0), name = proto.name)
                    DataType.FLOAT -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).float, type, dims = IntArray(0), name = proto.name)
                    DataType.INT64 -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).long, type, dims = IntArray(0), name = proto.name)
                    DataType.INT32 -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).int, type, dims = IntArray(0), name = proto.name)
                    DataType.INT16 -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).short, type, dims = IntArray(0), name = proto.name)
                    DataType.INT8 -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN), type, dims = IntArray(0), name = proto.name)
                    DataType.UINT8 -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN), type, dims = IntArray(0), name = proto.name)
                    DataType.BOOL -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).int != 0, type, dims = IntArray(0), name = proto.name)
                    else -> error("Unsupported data type $type")
                }
            } else Tensor(array[0], type, IntArray(0), proto.name)
        }
    }
}
