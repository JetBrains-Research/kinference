package io.kinference.utils

import io.kinference.data.tensors.Tensor
import io.kinference.protobuf.message.TensorProto
import java.io.File

object DataLoader {
    fun getTensor(file: File): Tensor = getTensor(file.readBytes())

    fun getTensor(bytes: ByteArray): Tensor = getTensor(TensorProto.decode(bytes))

    fun getTensor(tensorProto: TensorProto) = Tensor.create(tensorProto)
}
