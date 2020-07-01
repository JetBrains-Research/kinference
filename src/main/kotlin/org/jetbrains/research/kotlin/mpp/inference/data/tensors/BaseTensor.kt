package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXData
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXDataType
import org.jetbrains.research.kotlin.mpp.inference.types.TensorInfo

abstract class BaseTensor(info: TensorInfo) : ONNXData(ONNXDataType.ONNX_TENSOR, info) {
    //TODO: complex, uint32/64 tensors, strings

    abstract operator fun plus(other: Tensor): BaseTensor
    abstract operator fun plus(other: ScalarTensor): BaseTensor

    abstract operator fun times(other: Tensor): BaseTensor
    abstract operator fun times(other: ScalarTensor): BaseTensor

    abstract infix fun matmul(other: Tensor): BaseTensor
    abstract infix fun matmul(other: ScalarTensor): BaseTensor

    companion object {
        fun create(proto: TensorProto) = if (proto.dims.isNullOrEmpty()) ScalarTensor.create(proto) else Tensor.create(proto)
    }
}
