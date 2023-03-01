package io.kinference.tfjs.data.tensors

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.toNDArray
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.resolveProtoDataType
import io.kinference.types.*

fun <T : NDArrayTFJS> T.asTensor(name: String? = null) =
    TFJSTensor(this, ValueInfo(ValueTypeInfo.TensorTypeInfo(TensorShape(shape), type.resolveProtoDataType()), name ?: ""))

fun ArrayTFJS.asTensor(name: String? = null) = this.toNDArray().asTensor(name)

fun String.tfTypeResolve(): TensorProto.DataType {
    return when (this) {
        "float32" -> TensorProto.DataType.FLOAT
        "int32" -> TensorProto.DataType.INT32
        "bool" -> TensorProto.DataType.BOOL
        else -> error("Unsupported type $this")
    }
}

fun <T : NDArrayTFJS> Array<T>.asNamedOutputs(names: List<String>): List<TFJSTensor> {
    return List(this.size) {
        this[it].asTensor(names.getOrNull(it))
    }
}
