package org.jetbrains.research.kotlin.inference.extensions.tensor

import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.concatenate

fun Collection<Tensor>.stack(axis: Int): Tensor {
    return this.map {
        val newShape = this.first().data.shape.toMutableList()
        newShape.add(axis, 1)
        it.data.reshape(newShape.toIntArray())
    }.concatenate(axis).asTensor()
}

fun Collection<Tensor>.concatenate(axis: Int): Tensor {
    return this.reduce { acc, tensor -> acc.data.concatenate(tensor.data, axis).asTensor() }
}
