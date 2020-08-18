package org.jetbrains.research.kotlin.inference.data.tensors

import org.jetbrains.research.kotlin.inference.data.ndarray.MutableTypedNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArraysCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.matrixDotInto
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

fun unsqueezeFirst(shape: IntArray, newShapeSize: Int): IntArray {
    val wrapSize = newShapeSize - shape.size

    val wrappedShape = IntArray(newShapeSize)
    wrappedShape.fill(1, 0, wrapSize)
    shape.copyInto(wrappedShape, wrapSize)
    return wrappedShape
}

fun <T> TypedNDArray<T>.applyWithBroadcast(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>, op: PrimitiveArraysCombineFunction<T>, ordered: Boolean = false): MutableTypedNDArray<T> {
    val newShape = broadcastShape(this.shape, other.shape)

    require(destination.shape.contentEquals(newShape) && (!ordered || newShape.contentEquals(this.shape)))

    val leftShape = if (ordered) shape else unsqueezeFirst(shape, newShape.size)
    val rightShape = unsqueezeFirst(other.shape, newShape.size)

    val left = createNDArray(type, array, leftShape, offset)
    val right = createNDArray(type, other.array, rightShape, other.offset)

    broadcast(left, right, destination, op)
    return destination
}

fun <T> broadcast(left: TypedNDArray<T>, right: TypedNDArray<T>, destination: MutableTypedNDArray<T>, op: PrimitiveArraysCombineFunction<T>) {
    if (left.shape.contentEquals(right.shape)) {
        op.apply(left.array, left.offset, right.array, right.offset, destination.array, destination.offset, left.linearSize)
    } else {
        innerBroadcast(left, right, destination) { fstArray, sndArray, dest -> broadcast(fstArray, sndArray, dest, op) }
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
