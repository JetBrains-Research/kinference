package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXData
import org.jetbrains.research.kotlin.mpp.inference.types.TensorInfo
import org.jetbrains.research.kotlin.mpp.inference.types.TensorShape

class ScalarTensor(val value: Any, info: TensorInfo) : BaseTensor(info) {
    constructor(name: String?, value: Any, type: DataType) : this(value, TensorInfo(name ?: "", type, TensorShape.empty()))

    override fun plus(other: ScalarTensor): BaseTensor {
        val newValue = add(value as Number, other.value as Number)
        return ScalarTensor(newValue, info as TensorInfo)
    }

    override fun plus(other: Tensor): BaseTensor {
        return other + this
    }

    override fun times(other: ScalarTensor): BaseTensor {
        val newValue = times(value as Number, other.value as Number)
        return ScalarTensor(newValue, info as TensorInfo)
    }

    override fun times(other: Tensor): BaseTensor {
        return other * this
    }

    override fun matmul(other: ScalarTensor): BaseTensor {
        return this * other
    }

    override fun matmul(other: Tensor): BaseTensor {
        return other matmul this
    }

    override fun clone(newName: String): ONNXData = ScalarTensor(value, info as TensorInfo)

    companion object {
        fun create(proto: TensorProto): ScalarTensor = when (val type = DataType.fromValue(proto.data_type ?: 0)) {
            DataType.DOUBLE -> ScalarTensor(proto.name, proto.double_data[0], type)
            DataType.FLOAT -> ScalarTensor(proto.name, proto.float_data[0], type)
            DataType.INT64 -> ScalarTensor(proto.name, proto.int64_data[0], type)
            DataType.INT32 -> ScalarTensor(proto.name, proto.int32_data[0], type)
            else -> error("Unsupported data type")
        }
    }
}
