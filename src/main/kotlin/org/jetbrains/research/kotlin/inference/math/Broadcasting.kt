package org.jetbrains.research.kotlin.inference.math

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import kotlin.math.max

fun broadcastShape(firstShape: IntArray, secondShape: IntArray): IntArray {
    val totalShapeLength = max(firstShape.size, secondShape.size)

    return IntArray(totalShapeLength) { i ->
        val firstDim = firstShape.getOrNull(firstShape.size - i - 1) ?: 1
        val second = secondShape.getOrNull(secondShape.size - i - 1) ?: 1

        if (firstDim != second && firstDim != 1 && second != 1) error("Cannot broadcast shapes")

        max(firstDim, second)
    }.reversedArray()
}

// TODO remove to different module
fun unsqueezeFirst(shape: IntArray, newShapeSize: Int): IntArray {
    val wrapSize = newShapeSize - shape.size

    val wrappedShape = IntArray(newShapeSize)
    wrappedShape.fill(1, 0, wrapSize)
    shape.copyInto(wrappedShape, wrapSize)
    return wrappedShape
}

fun NDArray.applyWithBroadcast(other: NDArray, destination: MutableNDArray, ordered: Boolean = false, op: (NDArray, NDArray, MutableNDArray) -> Unit): MutableNDArray {
    val newShape = broadcastShape(this.shape, other.shape)

    require(destination.shape.contentEquals(newShape) && (!ordered || newShape.contentEquals(this.shape)))

    val leftShape = if (ordered) shape else unsqueezeFirst(shape, newShape.size)
    val rightShape = unsqueezeFirst(other.shape, newShape.size)

    val left = this.toMutable(Strides(leftShape))
    val right = other.toMutable(Strides(rightShape))

    broadcast(left, right, destination, op)
    return destination
}

fun broadcast(left: NDArray, right: NDArray, destination: MutableNDArray, op: (NDArray, NDArray, MutableNDArray) -> Unit) {
    if (left.shape.contentEquals(right.shape)) {
        op(left, right, destination)
    } else {
        innerBroadcast(left, right, destination) { fstArray, sndArray, dest -> broadcast(fstArray, sndArray, dest, op) }
    }
}

fun innerBroadcast(left: NDArray, right: NDArray, destination: MutableNDArray, recurrentBack: (NDArray, NDArray, MutableNDArray) -> Unit) {
    if (left.shape[0] != right.shape[0]) {
        val arrayWithOne = if (left.shape[0] == 1) left else right
        val arrayWithoutOne = if (left.shape[0] != 1) left else right

        val squeezedWithOne = arrayWithOne.view(0)

        for (i in 0 until arrayWithoutOne.shape[0]) {
            val movedWithoutOne = arrayWithoutOne.view(i)
            val movedOutput = destination.viewMutable(i)
            if (arrayWithOne == left) recurrentBack(squeezedWithOne, movedWithoutOne, movedOutput) else recurrentBack(movedWithoutOne, squeezedWithOne, movedOutput)
        }
    } else {
        for (i in 0 until left.shape[0]) {
            val movedLeft = left.view(i)
            val movedRight = right.view(i)
            val movedDestination = destination.viewMutable(i)
            recurrentBack(movedLeft, movedRight, movedDestination)
        }
    }
}
