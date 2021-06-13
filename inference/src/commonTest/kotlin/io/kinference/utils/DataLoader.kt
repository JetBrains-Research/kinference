package io.kinference.utils

import io.kinference.data.tensors.Tensor
import io.kinference.protobuf.message.TensorProto

object DataLoader {
    fun getTensor(bytes: ByteArray): Tensor = getTensor(TensorProto.decode(bytes))

    fun getTensor(tensorProto: TensorProto) = Tensor.create(tensorProto)
}
