package io.kinference.tfjs.data.tensors

import io.kinference.tfjs.custom_externals.core.TensorTFJS
import io.kinference.ndarray.arrays.NDArray
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.types.*

fun TensorTFJS.asTensor(name: String? = null) =
    Tensor(this, ValueInfo(ValueTypeInfo.TensorTypeInfo(TensorShape(this.shape.toIntArray()), dtype.tfTypeResolve()), name ?: ""))

fun NDArray.asTensor(name: String? = null) =
    Tensor(this, name)


fun String.tfTypeResolve(): TensorProto.DataType {
    return when (this) {
        "float32" -> TensorProto.DataType.FLOAT
        "int32" -> TensorProto.DataType.INT32
        "bool" -> TensorProto.DataType.BOOL
        else -> error("Unsupported type")
    }
}
