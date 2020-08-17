package org.jetbrains.research.kotlin.inference.data.tensors

import org.jetbrains.research.kotlin.inference.data.ndarray.MutableTypedNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArraysCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.concatenate
import org.jetbrains.research.kotlin.inference.extensions.ndarray.rows
import org.jetbrains.research.kotlin.inference.extensions.primitives.concat
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
