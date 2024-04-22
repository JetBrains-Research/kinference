package io.kinference.core.data.tensor

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.concat
import io.kinference.ndarray.extensions.splitWithAxis
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.resolveProtoDataType
import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo

fun NDArrayCore.asTensor(name: String? = null) = KITensor(name, this, ValueTypeInfo.TensorTypeInfo(TensorShape(this.shape), type.resolveProtoDataType()))

internal fun <T : NDArray> T.asTensor(name: String? = null) = (this as NDArrayCore).asTensor(name)

internal fun <T : NDArray> Collection<T>.asONNXTensors(names: List<String>): List<KITensor> {
    return this.zip(names).map { (data, name) -> data.asTensor(name) }
}

suspend fun Collection<KITensor>.stack(axis: Int): KITensor {
    val fstShape = this.first().data.shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)

    return this.map { it.data.reshape(newShape) }.concat(axis).asTensor()
}

suspend fun List<KITensor>.concatenate(axis: Int): KITensor {
    return this.map { it.data }.concat(axis).asTensor()
}

suspend fun KITensor.split(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<KITensor> {
    return data.splitWithAxis(parts, axis, keepDims).map { it.asTensor() }
}

suspend fun KITensor.split(split: IntArray, axis: Int = 0, keepDims: Boolean = true): List<KITensor> {
    return data.splitWithAxis(split, axis, keepDims).map { it.asTensor() }
}

suspend fun KITensor.split(splitTensor: KITensor, axis: Int = 0, keepDims: Boolean = true): List<KITensor> {
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

    return data.splitWithAxis(splitArray, axis, keepDims).map { it.asTensor() }
}
