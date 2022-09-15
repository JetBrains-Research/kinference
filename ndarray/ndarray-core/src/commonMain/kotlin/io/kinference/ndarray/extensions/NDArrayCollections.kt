package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*

fun Collection<NDArrayCore>.concat(axis: Int): MutableNDArrayCore {
    return this.first().concat(this.drop(1), axis)
}

fun Collection<NDArrayCore>.stack(axis: Int): MutableNDArrayCore {
    val fstShape = this.first().shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)
    return this.map { it.reshape(newShape) }.concat(axis)
}
