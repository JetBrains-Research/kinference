package io.kinference.core.data.tensor

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.concatenate
import io.kinference.ndarray.extensions.splitWithAxis
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.resolveProtoDataType
import io.kinference.core.types.*

fun NDArray.asTensor(name: String? = null) = KITensor(this, ValueInfo(ValueTypeInfo.TensorTypeInfo(TensorShape(this.shape), type.resolveProtoDataType()), name ?: ""))

fun Collection<KITensor>.stack(axis: Int): KITensor {
    val fstShape = this.first().data.shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)

    return this.map { it.data.reshapeView(newShape) }.concatenate(axis).asTensor()
}

fun List<KITensor>.concatenate(axis: Int): KITensor {
    return this.map { it.data }.concatenate(axis).asTensor()
}

fun KITensor.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<KITensor> {
    return data.splitWithAxis(parts, axis, keepDims).map { it.asTensor() }
}

fun KITensor.splitWithAxis(split: IntArray, axis: Int = 0, keepDims: Boolean = true): List<KITensor> {
    return data.splitWithAxis(split, axis, keepDims).map { it.asTensor() }
}


fun KITensor.splitWithAxis(splitTensor: KITensor, axis: Int = 0, keepDims: Boolean = true): List<KITensor> {
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
