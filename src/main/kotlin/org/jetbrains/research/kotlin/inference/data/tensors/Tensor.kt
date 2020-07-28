package org.jetbrains.research.kotlin.inference.data.tensors

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.ONNXDataType
import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.createArray
import org.jetbrains.research.kotlin.inference.extensions.primitives.toIntArray
import org.jetbrains.research.kotlin.inference.types.TensorInfo
import org.jetbrains.research.kotlin.inference.types.TensorShape

//TODO: support segments
//TODO: support external and raw data
@Suppress("UNCHECKED_CAST")
class Tensor(val data: NDArray, info: TensorInfo) : ONNXData(ONNXDataType.ONNX_TENSOR, info) {
    override fun clone(newName: String): ONNXData {
        return Tensor(data.clone(), TensorInfo(newName, data.type, TensorShape(data.shape)))
    }

    fun mapElements(type: DataType = this.data.type, func: (Any) -> Any): Tensor {
        val buffer = createArray(type, data.linearSize) { func(data[it]) }
        return NDArray(buffer, type, data.shape).asTensor()
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
            if (proto.dims.isNullOrEmpty()) return createScalar(proto)

            return when (val type = DataType.fromValue(proto.data_type ?: 0)) {
                DataType.DOUBLE -> Tensor(proto.double_data.toDoubleArray(), type, proto.dims.toIntArray(), proto.name)
                DataType.FLOAT -> Tensor(proto.float_data.toFloatArray(), type, proto.dims.toIntArray(), proto.name)
                DataType.INT64 -> Tensor(proto.int64_data.toLongArray(), type, proto.dims.toIntArray(), proto.name)
                DataType.INT32 -> Tensor(proto.int32_data.toIntArray(), type, proto.dims.toIntArray(), proto.name)
                DataType.STRING -> Tensor(proto.string_data.map { it.utf8() }, type, proto.dims.toIntArray(), proto.name)
                else -> error("Unsupported data type $type")
            }
        }

        operator fun invoke(dims: List<Long>, value: List<*>, type: DataType, name: String?): Tensor {
            val data = createArray(type, value.size) { i -> value[i]!! }
            return Tensor(data, type, dims.toIntArray(), name)
        }


        operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0), name: String? = ""): Tensor {
            return when (value) {
                is DoubleArray -> DoubleNDArray(value, Strides(dims)).asTensor(name!!)
                is FloatArray -> FloatNDArray(value, Strides(dims)).asTensor(name!!)
                is IntArray -> IntNDArray(value, Strides(dims)).asTensor(name!!)
                is LongArray -> LongNDArray(value, Strides(dims)).asTensor(name!!)
                is ShortArray -> ShortNDArray(value, Strides(dims)).asTensor(name!!)
                is Boolean -> BooleanNDArray(BooleanArray(1) { value }, Strides(dims)).asTensor(name!!)
                is Long -> LongNDArray(LongArray(1) { value }, Strides(dims)).asTensor(name!!)
                else -> error("Unsupported data type $type")
            }
        }

        operator fun invoke(value: List<Any>, type: DataType): Tensor {
            val dims = intArrayOf(value.size)
            val data = createArray(type, value.size) { i -> value[i] }
            return Tensor(data, type, dims)
        }

        private fun createScalar(proto: TensorProto): Tensor {
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
                    DataType.DOUBLE -> Tensor(proto.raw_data!!.asByteBuffer().double, type, dims = IntArray(0), name = proto.name)
                    DataType.FLOAT -> Tensor(proto.raw_data!!.asByteBuffer().float, type, dims = IntArray(0), name = proto.name)
                    DataType.INT64 -> Tensor(proto.raw_data!!.asByteBuffer().long, type, dims = IntArray(0), name = proto.name)
                    DataType.INT32 -> Tensor(proto.raw_data!!.asByteBuffer().int, type, dims = IntArray(0), name = proto.name)
                    DataType.BOOL -> Tensor(proto.raw_data!!.asByteBuffer().int != 0, type, dims = IntArray(0), name = proto.name)
                    else -> error("Unsupported data type")
                }
            } else Tensor(array[0], type, IntArray(0), proto.name)
        }
    }
}
