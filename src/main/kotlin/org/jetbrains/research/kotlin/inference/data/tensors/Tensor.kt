package org.jetbrains.research.kotlin.inference.data.tensors


import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.ONNXDataType
import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.toIntArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import org.jetbrains.research.kotlin.inference.types.TensorInfo
import org.jetbrains.research.kotlin.inference.types.TensorShape
import java.nio.ByteBuffer
import java.nio.ByteOrder

//TODO: support segments
//TODO: support external and raw data
@Suppress("UNCHECKED_CAST")
class Tensor(val data: TypedNDArray<Any>, info: TensorInfo) : ONNXData(ONNXDataType.ONNX_TENSOR, info) {
    override fun clone(newName: String): ONNXData {
        return Tensor(data.clone(), TensorInfo(newName, data.type, TensorShape(data.shape)))
    }

    override fun rename(newName: String): ONNXData {
        return Tensor(data, TensorInfo(newName, data.type, TensorShape(data.shape)))
    }

    fun mapElements(type: DataType = this.data.type, func: (Any) -> Any): Tensor {
        val buffer = createArray(type, data.linearSize) { func(data[it]) }
        return createNDArray(type, buffer, data.shape).asTensor()
    }

    operator fun plus(other: Tensor): Tensor {
        return (this.data + other.data).asTensor()
    }

    operator fun times(other: Tensor): Tensor {
        return (this.data * other.data).asTensor()
    }

    infix fun matmul(other: Tensor): Tensor {
        return (this.data matmul other.data).asTensor()
    }

    companion object {
        //TODO: complex, uint32/64 tensors
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
                else -> error("Unsupported data type")
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
                    DataType.BOOL -> Tensor(BooleanArray(sizeInBytes) { buffer[it] != 0.toByte() }, type, shape, proto.name)
                    DataType.STRING -> error("String data MUST not be present in raw_data field")
                    else -> error("Unsupported data type")
                }
            } else when (type) {
                DataType.DOUBLE -> Tensor((array as List<Double>).toDoubleArray(), type, shape, proto.name)
                DataType.FLOAT -> Tensor((array as List<Float>).toFloatArray(), type, shape, proto.name)
                DataType.INT64 -> Tensor((array as List<Long>).toLongArray(), type, shape, proto.name)
                DataType.INT32 -> Tensor((array as List<Int>).toIntArray(), type, shape, proto.name)
                DataType.BOOL -> Tensor((array as List<Boolean>).toBooleanArray(), type, shape, proto.name)
                DataType.STRING -> Tensor((array as List<String>).toTypedArray(), type, shape, proto.name)
                else -> error("Unsupported data type")
            }
        }

        operator fun invoke(dims: List<Long>, value: List<*>, type: DataType, name: String?): Tensor {
            val data = createArray(type, value.size) { i -> value[i]!! }
            return Tensor(data, type, dims.toIntArray(), name)
        }


        operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0), name: String? = ""): Tensor {
            if (dims.isEmpty()) return createScalarNDArray<Any>(type, value).asTensor(name)

            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(value as DoubleArray, Strides(dims)).asTensor(name!!)
                DataType.FLOAT -> FloatNDArray(value as FloatArray, Strides(dims)).asTensor(name!!)
                DataType.INT32 -> IntNDArray(value as IntArray, Strides(dims)).asTensor(name!!)
                DataType.INT64 -> LongNDArray(value as LongArray, Strides(dims)).asTensor(name!!)
                DataType.INT16 -> ShortNDArray(value as ShortArray, Strides(dims)).asTensor(name!!)
                DataType.BOOL -> BooleanNDArray(value as BooleanArray, Strides(dims)).asTensor(name!!)
                else -> error("Unsupported data type $type")
            }
        }

        operator fun invoke(value: List<Any>, type: DataType): Tensor {
            val dims = intArrayOf(value.size)
            val data = createArray(type, value.size) { i -> value[i] }
            return Tensor(data, type, dims)
        }

        private fun parseScalar(proto: TensorProto): Tensor {
            val type = DataType.fromValue(proto.data_type ?: 0)
            val array = when (type) {
                DataType.DOUBLE -> proto.double_data
                DataType.FLOAT -> proto.float_data
                DataType.INT64 -> proto.int64_data
                DataType.INT32 -> proto.int32_data
                DataType.BOOL -> proto.int32_data.map { it != 0 }
                else -> error("Unsupported data type")
            }
            return if (array.isEmpty()) {
                when (type) {
                    DataType.DOUBLE -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).double, type, dims = IntArray(0), name = proto.name)
                    DataType.FLOAT -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).float, type, dims = IntArray(0), name = proto.name)
                    DataType.INT64 -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).long, type, dims = IntArray(0), name = proto.name)
                    DataType.INT32 -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).int, type, dims = IntArray(0), name = proto.name)
                    DataType.INT16 -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).short, type, dims = IntArray(0), name = proto.name)
                    DataType.BOOL -> Tensor(proto.raw_data!!.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).int != 0, type, dims = IntArray(0), name = proto.name)
                    else -> error("Unsupported data type")
                }
            } else Tensor(array[0], type, IntArray(0), proto.name)
        }
    }
}
