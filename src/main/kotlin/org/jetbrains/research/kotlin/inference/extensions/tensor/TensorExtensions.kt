package org.jetbrains.research.kotlin.inference.extensions.tensor

import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.concatenate
import org.jetbrains.research.kotlin.inference.extensions.ndarray.splitWithAxis

fun Collection<Tensor>.stack(axis: Int): Tensor {
    val fstShape = this.first().data.shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)

    return this.map { it.data.reshape(newShape) }.concatenate(axis).asTensor()
}

fun Collection<Tensor>.concatenate(axis: Int): Tensor {
    return this.reduce { acc, tensor -> acc.data.concatenate(tensor.data, axis).asTensor() }
}

fun Tensor.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
    return data.splitWithAxis(parts, axis, keepDims).map { it.asTensor() }
}

fun Tensor.splitWithAxis(split: IntArray, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
    return data.splitWithAxis(split, axis, keepDims).map { it.asTensor() }
}

fun Tensor.splitWithAxis(splitTensor: Tensor, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
    return this.data.splitWithAxis(splitTensor.data.array as IntArray, axis, keepDims).map { it.asTensor() }
}
