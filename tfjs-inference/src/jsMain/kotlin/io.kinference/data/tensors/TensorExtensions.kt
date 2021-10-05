package io.kinference.data.tensors

import io.kinference.custom_externals.core.TensorTFJS
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.concatenate
import io.kinference.ndarray.extensions.splitWithAxis
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.resolveProtoDataType
import io.kinference.types.*

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
