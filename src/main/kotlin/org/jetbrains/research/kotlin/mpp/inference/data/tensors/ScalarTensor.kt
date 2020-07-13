package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXData
import org.jetbrains.research.kotlin.mpp.inference.types.TensorInfo
import org.jetbrains.research.kotlin.mpp.inference.types.TensorShape

class ScalarTensor(val value: Any, info: TensorInfo) : BaseTensor(info) {
    constructor(name: String?, value: Any, type: DataType) : this(value, TensorInfo(name ?: "", type, TensorShape.empty()))

    override fun plus(other: BaseTensor): BaseTensor {
        return when (other) {
            is ScalarTensor -> ScalarTensor(add(value as Number, other.value as Number), info as TensorInfo)
            is Tensor -> other + this
            else -> error("Unknown tensor type")
        }
    }

    override fun times(other: BaseTensor): BaseTensor {
        return when (other) {
            is ScalarTensor -> ScalarTensor(times(value as Number, other.value as Number), info as TensorInfo)
            is Tensor -> other * this
            else -> error("Unknown tensor type")
        }
    }

    override fun matmul(other: BaseTensor): BaseTensor {
        return when (other) {
            is ScalarTensor -> this * other
            is Tensor -> other matmul this
            else -> error("Unknown tensor type")
        }
    }

    override fun clone(newName: String): ONNXData = ScalarTensor(value, TensorInfo(newName, info.type, (info as TensorInfo).shape))

    companion object {
        fun create(proto: TensorProto): ScalarTensor {
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
                    DataType.DOUBLE -> ScalarTensor(proto.name, proto.raw_data!!.asByteBuffer().double, type)
                    DataType.FLOAT -> ScalarTensor(proto.name, proto.raw_data!!.asByteBuffer().float, type)
                    DataType.INT64 -> ScalarTensor(proto.name, proto.raw_data!!.asByteBuffer().long, type)
                    DataType.INT32 -> ScalarTensor(proto.name, proto.raw_data!!.asByteBuffer().int, type)
                    DataType.BOOL -> ScalarTensor(proto.name, proto.raw_data!!.asByteBuffer().int != 0, type)
                    else -> error("Unsupported data type")
                }
            } else ScalarTensor(proto.name, array[0], type)
        }
    }
}
