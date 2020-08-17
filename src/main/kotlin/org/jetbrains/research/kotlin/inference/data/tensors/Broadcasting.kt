package org.jetbrains.research.kotlin.inference.data.tensors

import org.jetbrains.research.kotlin.inference.data.ndarray.MutableTypedNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArraysCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
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

fun <T> TypedNDArray<T>.broadcast(newShape: IntArray): TypedNDArray<T> {
    if (this.shape.contentEquals(newShape)) return this

    val unsqueezedShape = if (newShape.size <= rank) {
        this.shape.copyOf()
    } else {
        unsqueezeFirst(this.shape, newShape.size)
    }

    val unsqueezedStrides = Strides(unsqueezedShape)
    val newStrides = Strides(newShape)
    val newArray = allocateNDArray<T>(type, newStrides)

    for (i in newStrides.strides.size - 1 until 0) {
        val blockSize = unsqueezedStrides.strides[i]
        val requiredBlockSize = newStrides.strides[i]
        if (blockSize < requiredBlockSize) {
            for (j in 0 until (requiredBlockSize / blockSize)) {
                newArray.place(blockSize * j, array, 0, blockSize)
            }
        }
    }
    return newArray
}

fun unsqueezeFirst(shape: IntArray, newShapeSize: Int): IntArray {
    val wrapSize = newShapeSize - shape.size

    val wrappedShape = IntArray(newShapeSize)
    wrappedShape.fill(1, 0, wrapSize)
    shape.copyInto(wrappedShape, wrapSize)
    return wrappedShape
}

fun <T> TypedNDArray<T>.applyWithBroadcast(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>, op: PrimitiveArraysCombineFunction<T>) {
    val newShape = broadcastShape(this.shape, other.shape)

    require(destination.shape.contentEquals(newShape))

    val wrappedLeftShape = unsqueezeFirst(shape, newShape.size)
    val wrappedRightShape = unsqueezeFirst(other.shape, newShape.size)

    val wrappedLeft = createNDArray(type, array, wrappedLeftShape, offset)
    val wrappedRight = createNDArray(type, other.array, wrappedRightShape, other.offset)


    broadcast(wrappedLeft, wrappedRight, destination, op)
}

fun <T> broadcast(left: TypedNDArray<T>, right: TypedNDArray<T>, destination: MutableTypedNDArray<T>, op: PrimitiveArraysCombineFunction<T>) {
    if (left.shape.contentEquals(right.shape)) {
        op.apply(left.array, left.offset, right.array, right.offset, destination.array, destination.offset, left.linearSize)
    } else {
        innerBroadcast(left, right, destination) { left, right, destination -> broadcast(left, right, destination, op) }
    }
}

fun <T> broadcastDot(left: TypedNDArray<T>, right: TypedNDArray<T>, destination: MutableTypedNDArray<T>) {
    if (left.shape.size == 2) {
        left.matrixDotInto(right, destination, false)
    } else {
        innerBroadcast(left, right, destination, ::broadcastDot)
    }
}

fun <T> innerBroadcast(left: TypedNDArray<T>, right: TypedNDArray<T>, destination: MutableTypedNDArray<T>, recurrentBack: (TypedNDArray<T>, TypedNDArray<T>, MutableTypedNDArray<T>) -> Unit) {
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
