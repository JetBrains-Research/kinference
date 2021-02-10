package io.kinference.utils

import io.kinference.data.tensors.Tensor
import io.kinference.onnx.TensorProto
import java.io.File

object DataLoader {
    fun getTensor(bytes: ByteArray): Tensor = getTensor(TensorProto.ADAPTER.decode(bytes))

    fun getTensor(tensorProto: TensorProto) = Tensor.create(tensorProto)
}
