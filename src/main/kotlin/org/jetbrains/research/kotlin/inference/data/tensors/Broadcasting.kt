package org.jetbrains.research.kotlin.inference.data.tensors

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.primitives.matrixDotInto
import kotlin.math.max

fun broadcastShape(currentShape: IntArray, newShape: IntArray): IntArray {
    val totalShapeLength = max(currentShape.size, newShape.size)

    return IntArray(totalShapeLength) { i ->
        val currentDim = currentShape.getOrNull(currentShape.size - i - 1) ?: 1
        val newDim = newShape.getOrNull(newShape.size - i - 1) ?: 1

        if (currentDim != newDim && currentDim != 1 && newDim != 1) error("Cannot broadcast shapes")

        max(currentDim, newDim)
    }.reversedArray()
}

fun <T> NDArray<T>.applyWithBroadcast(other: NDArray<T>, destination: NDArray<T>?, op: PrimitiveArrayCombineFunction<T>): NDArray<T> {
    val newShape = broadcastShape(this.shape, other.shape)
    val wrapSizeLeft = newShape.size - shape.size
    val wrapSizeRight = newShape.size - other.shape.size
    val wrappedLeft = this.unsqueeze(*IntArray(wrapSizeLeft) { it })
    val wrappedRight = other.unsqueeze(*IntArray(wrapSizeRight) { it })

    val actualDestination = destination ?: allocateNDArray(type, Strides(newShape)) as NDArray<T>
    require(actualDestination.shape.contentEquals(newShape))

    broadcast(wrappedLeft, wrappedRight, actualDestination, op)
    return actualDestination
    //return NDArray(op.apply(castedThis.array, castedThis.offset, castedOther.array, castedOther.offset, actualDestination.array, actualDestination.offset, castedThis.linearSize), type, newShape)
}

fun <T> broadcast(left: NDArray<T>, right: NDArray<T>, destination: NDArray<T>, op: PrimitiveArrayCombineFunction<T>) {
    if (left.shape.contentEquals(right.shape)) {
        op.apply(left.array, left.offset, right.array, right.offset, destination.array, destination.offset, left.linearSize)
    } else {
        innerBroadcast(left, right, destination) { left, right, destination -> broadcast(left, right, destination, op) }
    }
}

fun <T> broadcastDot(left: NDArray<T>, right: NDArray<T>, destination: NDArray<T>) {
    if (left.shape.size == 2) {
        left.matrixDotInto(right, destination, false)
    } else {
        innerBroadcast(left, right, destination, ::broadcastDot)
    }
}

fun <T> innerBroadcast(left: NDArray<T>, right: NDArray<T>, destination: NDArray<T>, recurrentBack: (NDArray<T>, NDArray<T>, NDArray<T>) -> Unit) {
    if (left.shape[0] != right.shape[0]) {
        val arrayWithOne = if (left.shape[0] == 1) left else right
        val arrayWithoutOne = if (left.shape[0] != 1) left else right

        val newStridesWithOne = Strides(arrayWithOne.shape.copyOfRange(1, arrayWithOne.shape.size))
        val newStridesWithoutOne = Strides(arrayWithoutOne.shape.copyOfRange(1, arrayWithoutOne.shape.size))
        val newDestinationStrides = Strides(destination.shape.copyOfRange(1, destination.shape.size))

        val blockSizeWithoutOne = arrayWithoutOne.strides.strides[0]
        val blockSizeDestination = destination.strides.strides[0]

        val squeezedWithOne = arrayWithOne.move(0, newStridesWithOne)

        for (i in 0 until arrayWithoutOne.shape[0]) {
            val movedWithoutOne = arrayWithoutOne.move(blockSizeWithoutOne * i, newStridesWithoutOne)
            val movedOutput = destination.move(blockSizeDestination * i, newDestinationStrides)
            if (arrayWithOne == left) recurrentBack(squeezedWithOne, movedWithoutOne, movedOutput) else recurrentBack(movedWithoutOne, squeezedWithOne, movedOutput)
        }
    } else {
        val newStridesLeft = Strides(left.shape.copyOfRange(1, left.shape.size))
        val newStridesRight = Strides(right.shape.copyOfRange(1, right.shape.size))
        val newStridesDestination = Strides(destination.shape.copyOfRange(1, destination.shape.size))

        val leftBlockSize = left.strides.strides[0]
        val rightBlockSize = right.strides.strides[0]
        val destinationBlockSize = destination.strides.strides[0]

        for (i in 0 until left.shape[0]) {
            val movedLeft = left.move(leftBlockSize * i, newStridesLeft)
            val movedRight = right.move(rightBlockSize * i, newStridesRight)
            val movedDestination = destination.move(destinationBlockSize * i, newStridesDestination)
            recurrentBack(movedLeft, movedRight, movedDestination)
        }
    }
}
