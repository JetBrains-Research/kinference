package org.jetbrains.research.kotlin.inference.data.tensors

import org.jetbrains.research.kotlin.inference.math.asTensor
import org.jetbrains.research.kotlin.inference.math.concatenate
import org.jetbrains.research.kotlin.inference.math.extensions.splitWithAxis

fun Collection<Tensor>.stack(axis: Int): Tensor {
    val fstShape = this.first().data.shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)

    return this.map { it.data.toMutable().reshape(newShape) }.concatenate(axis).asTensor()
}

fun List<Tensor>.concatenate(axis: Int): Tensor {
    var acc = this[0].data.toMutable()
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
    val splitArray = IntArray(splitTensor.data.linearSize) { i -> (splitTensor.data[i] as Number).toInt() }
    return this.data.splitWithAxis(splitArray, axis, keepDims).map { it.asTensor() }
}
