package io.kinference.tfjs.data.tensors

import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.externals.core.NDArrayTFJS
import io.kinference.tfjs.types.*

fun NDArrayTFJS.asTensor(name: String? = null) =
    TFJSTensor(this, ValueInfo(ValueTypeInfo.TensorTypeInfo(TensorShape(shape.toIntArray()), dtype.tfTypeResolve()), name ?: ""))

fun String.tfTypeResolve(): TensorProto.DataType {
    return when (this) {
        "float32" -> TensorProto.DataType.FLOAT
        "int32" -> TensorProto.DataType.INT32
        "bool" -> TensorProto.DataType.BOOL
        else -> error("Unsupported type")
    }
}
