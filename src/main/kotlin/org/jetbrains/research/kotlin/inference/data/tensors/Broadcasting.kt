package org.jetbrains.research.kotlin.inference.data.tensors

import org.jetbrains.research.kotlin.inference.data.ndarray.MutableTypedNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArraysCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.concatenate
import org.jetbrains.research.kotlin.inference.extensions.ndarray.createNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.rows
import org.jetbrains.research.kotlin.inference.extensions.primitives.concat
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

/*
fun broadcastMatrixElementsShape(fstShape: IntArray, sndShape: IntArray): Pair<IntArray, IntArray> {
    val base = broadcastShape(fstShape.copyOfRange(0, fstShape.size - 2), sndShape.copyOfRange(0, sndShape.size - 2))

    val first = IntArray(base.size + 2).apply {
        base.copyInto(this)
        fstShape.copyInto(this, base.size, fstShape.size - 2)
    }
    val second = IntArray(base.size + 2).apply {
        base.copyInto(this)
        sndShape.copyInto(this, base.size, sndShape.size - 2)
    }
    return first to second
}

fun <T> MutableTypedNDArray<T>.innerBroadcast(newShape: IntArray, asMatrixStack: Boolean = false): MutableTypedNDArray<T> {
    if (this.shape.contentEquals(newShape) || asMatrixStack && this.rank <= 2) return this

    val castShape = newShape.copyOfRange(1, newShape.size)

    //broadcast is available only if corresponding dims are equal or at least one of them is 1
    return when (this.shape[0]) {
        1 -> {
            val rows = (this.row(0) as MutableTypedNDArray<T>).innerBroadcast(castShape)
            val times = newShape[0]
            val reshaped = rows.reshape(1.concat(castShape))
            val resShape = reshaped.shape.copyOf().apply { set(0, times) }

            allocateNDArray<T>(type, Strides(resShape)).apply {
                for (i in 0 until times) {
                    placeAll(i * reshaped.linearSize, reshaped.array)
                }
            }
        }
        newShape[0] -> this.rows.map { it.innerBroadcast(castShape) }.concatenate(axis = 0).toMutable()
        else -> error("Cannot broadcast tensors")
    }
}

fun <T> TypedNDArray<T>.broadcast(newShape: IntArray, asMatrixStack: Boolean = false): TypedNDArray<T> {
    if (this.shape.contentEquals(newShape)) return this

    val newDims = if (newShape.size <= rank) {
        this.shape.copyOf()
    } else {
        val offset = newShape.size - this.rank
        IntArray(newShape.size).apply {
            this@broadcast.shape.copyInto(this, offset)
            fill(1, 0, offset)
        }
    }
    val preResult = this.toMutable().reshape(newDims)

    return preResult.innerBroadcast(newShape, asMatrixStack)
}


fun <T : Any> TypedNDArray<T>.applyWithBroadcast(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>, op: PrimitiveArraysCombineFunction<T>) {
    val newShape = broadcastShape(this.shape, other.shape)
    val castedThis = this.broadcast(newShape)
    val castedOther = other.broadcast(newShape)

    require(newShape.contentEquals(destination.shape))
    op.apply(castedThis.array, castedThis.offset, castedOther.array, castedOther.offset, destination.array, destination.offset, castedThis.linearSize)

    //return destination
}
*/

fun <T> MutableTypedNDArray<T>.innerBroadcast(newShape: IntArray, asMatrixStack: Boolean = false): MutableTypedNDArray<T> {
    if (this.shape.contentEquals(newShape) || asMatrixStack && this.rank <= 2) return this

    val castShape = newShape.copyOfRange(1, newShape.size)

    //broadcast is available only if corresponding dims are equal or at least one of them is 1
    return when (this.shape[0]) {
        1 -> {
            val rows = (this.row(0) as MutableTypedNDArray<T>).innerBroadcast(castShape)
            val times = newShape[0]
            val reshaped = rows.reshape(1.concat(castShape))
            val resShape = reshaped.shape.copyOf().apply { set(0, times) }

            allocateNDArray<T>(type, Strides(resShape)).apply {
                for (i in 0 until times) {
                    placeAll(i * reshaped.linearSize, reshaped.array)
                }
            }
        }
        newShape[0] -> this.rows.map { it.innerBroadcast(castShape) }.concatenate(axis = 0).toMutable()
        else -> error("Cannot broadcast tensors")
    }
}

fun <T> TypedNDArray<T>.broadcast(newShape: IntArray, asMatrixStack: Boolean = false): TypedNDArray<T> {
    if (this.shape.contentEquals(newShape)) return this

    val newDims = if (newShape.size <= rank) {
        this.shape.copyOf()
    } else {
        val offset = newShape.size - this.rank
        IntArray(newShape.size).apply {
            this@broadcast.shape.copyInto(this, offset)
            fill(1, 0, offset)
        }
    }
    val preResult = this.toMutable().reshape(newDims)

    return preResult.innerBroadcast(newShape, asMatrixStack)
}

fun wrapOnes(shape: IntArray, newShapeSize: Int): IntArray {
    val wrapSize = newShapeSize - shape.size

    val wrappedShape = IntArray(newShapeSize)
    wrappedShape.fill(1, 0, wrapSize)
    shape.copyInto(wrappedShape, wrapSize)
    return wrappedShape
}

fun <T> TypedNDArray<T>.applyWithBroadcast(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>, op: PrimitiveArraysCombineFunction<T>) {
    val newShape = broadcastShape(this.shape, other.shape)

    require(destination.shape.contentEquals(newShape))

    val wrappedLeftShape = wrapOnes(shape, newShape.size)
    val wrappedRightShape = wrapOnes(other.shape, newShape.size)

    val wrappedLeft = createNDArray(type, array, wrappedLeftShape, offset)
    val wrappedRight = createNDArray(type, other.array, wrappedRightShape, other.offset)


    broadcast(wrappedLeft, wrappedRight, destination, op)
    //return actualDestination
    //return NDArray(op.apply(castedThis.array, castedThis.offset, castedOther.array, castedOther.offset, actualDestination.array, actualDestination.offset, castedThis.linearSize), type, newShape)
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

        //val newStridesWithOne = Strides(arrayWithOne.shape.copyOfRange(1, arrayWithOne.shape.size))
        //val newStridesWithoutOne = Strides(arrayWithoutOne.shape.copyOfRange(1, arrayWithoutOne.shape.size))
        //val newDestinationStrides = Strides(destination.shape.copyOfRange(1, destination.shape.size))

        //val blockSizeWithoutOne = arrayWithoutOne.strides.strides[0]
        //val blockSizeDestination = destination.strides.strides[0]

        val squeezedWithOne = arrayWithOne.view(0)

        for (i in 0 until arrayWithoutOne.shape[0]) {
            val movedWithoutOne = arrayWithoutOne.view(i)
            val movedOutput = destination.viewMutable(i)
            if (arrayWithOne == left) recurrentBack(squeezedWithOne, movedWithoutOne, movedOutput) else recurrentBack(movedWithoutOne, squeezedWithOne, movedOutput)
        }
    } else {
        //val newStridesLeft = Strides(left.shape.copyOfRange(1, left.shape.size))
        //val newStridesRight = Strides(right.shape.copyOfRange(1, right.shape.size))
        //val newStridesDestination = Strides(destination.shape.copyOfRange(1, destination.shape.size))

        //val leftBlockSize = left.strides.strides[0]
        //val rightBlockSize = right.strides.strides[0]
        //val destinationBlockSize = destination.strides.strides[0]

        for (i in 0 until left.shape[0]) {
            val movedLeft = left.view(i)
            val movedRight = right.view(i)
            val movedDestination = destination.viewMutable(i)
            recurrentBack(movedLeft, movedRight, movedDestination)
        }
    }
}
