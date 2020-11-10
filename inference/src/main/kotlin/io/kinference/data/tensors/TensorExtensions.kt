package io.kinference.data.tensors

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.concatenate
import io.kinference.ndarray.extensions.splitWithAxis
import io.kinference.onnx.TensorProto
import io.kinference.primitives.types.DataType
import io.kinference.types.TensorInfo
import io.kinference.types.TensorShape

fun TensorProto.DataType.resolveLocalDataType(): DataType {
    return when(this) {
        TensorProto.DataType.DOUBLE -> DataType.DOUBLE
        TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16 -> DataType.FLOAT
        TensorProto.DataType.INT32 -> DataType.INT
        TensorProto.DataType.INT64 -> DataType.LONG
        TensorProto.DataType.INT16 -> DataType.SHORT
        TensorProto.DataType.INT8-> DataType.BYTE
        TensorProto.DataType.BOOL -> DataType.BOOLEAN
        TensorProto.DataType.UINT32-> DataType.UINT
        TensorProto.DataType.UINT64 -> DataType.ULONG
        TensorProto.DataType.UINT16 -> DataType.USHORT
        TensorProto.DataType.UINT8 -> DataType.UBYTE
        TensorProto.DataType.UNDEFINED -> DataType.UNKNOWN
        else -> error("Cannot resolve data type")
    }
}

fun DataType.resolveProtoDataType(): TensorProto.DataType {
    return when(this) {
        DataType.DOUBLE -> TensorProto.DataType.DOUBLE
        DataType.FLOAT -> TensorProto.DataType.FLOAT
        DataType.INT -> TensorProto.DataType.INT32
        DataType.LONG -> TensorProto.DataType.INT64
        DataType.SHORT -> TensorProto.DataType.INT16
        DataType.BYTE -> TensorProto.DataType.INT8
        DataType.BOOLEAN -> TensorProto.DataType.BOOL
        DataType.UINT -> TensorProto.DataType.UINT32
        DataType.ULONG -> TensorProto.DataType.UINT64
        DataType.USHORT -> TensorProto.DataType.UINT16
        DataType.UBYTE -> TensorProto.DataType.UINT8
        DataType.UNKNOWN -> TensorProto.DataType.UNDEFINED
        else -> error("Cannot resolve data type")
    }
}

fun NDArray.asTensor(name: String? = null) = Tensor(this, TensorInfo(name ?: "", type.resolveProtoDataType(), TensorShape(this.shape)))

fun Collection<Tensor>.stack(axis: Int): Tensor {
    val fstShape = this.first().data.shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)

    return this.map { it.data.reshapeView(newShape) }.concatenate(axis).asTensor()
}

fun List<Tensor>.concatenate(axis: Int): Tensor {
    var acc = this[0].data
    for (i in 1 until this.size) {
        acc = acc.concatenate(this[i].data, axis)
    }
    return acc.asTensor()
}

fun Tensor.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
    return data.splitWithAxis(parts, axis, keepDims).map { it.asTensor() }
}

fun Tensor.splitWithAxis(split: IntArray, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
    return data.splitWithAxis(split, axis, keepDims).map { it.asTensor() }
}


fun Tensor.splitWithAxis(splitTensor: Tensor, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
    val splitArray = when (splitTensor.data.type) {
        DataType.INT -> {
            val array = splitTensor.data as IntNDArray
            val pointer = array.array.pointer()
            IntArray(splitTensor.data.linearSize) { pointer.getAndIncrement() }
        }
        DataType.LONG -> {
            val array = splitTensor.data as LongNDArray
            val pointer = array.array.pointer()
            IntArray(splitTensor.data.linearSize) { pointer.getAndIncrement().toInt() }
        }
        else -> throw IllegalStateException("Split tensor must have Int or Long type")
    }

    return this.data.splitWithAxis(splitArray, axis, keepDims).map { it.asTensor() }
}
