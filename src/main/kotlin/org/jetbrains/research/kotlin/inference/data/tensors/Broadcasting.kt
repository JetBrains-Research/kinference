package org.jetbrains.research.kotlin.inference.data.tensors

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.concatenate
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

    val fst = IntArray(base.size + 2).apply {
        base.copyInto(this)
        fstShape.copyInto(this, base.size, fstShape.size - 2)
    }
    val snd = IntArray(base.size + 2).apply {
        base.copyInto(this)
        sndShape.copyInto(this, base.size, sndShape.size - 2)
    }
    return fst to snd
}

fun <T> NDArray<T>.innerBroadcast(newShape: IntArray, asMatrixStack: Boolean = false): NDArray<T> {
    if (this.shape.contentEquals(newShape) || asMatrixStack && this.rank <= 2) return this

    val castShape = newShape.copyOfRange(1, newShape.size)

    //broadcast is available only if corresponding dims are equal or at least one of them is 1
    return when (this.shape[0]) {
        1 -> {
            val rows = this.row(0).innerBroadcast(castShape)
            rows.reshape(intArrayOf(1, *castShape)).repeatRow(newShape[0])
        }
        newShape[0] -> this.rows.map { it.innerBroadcast(castShape) }.concatenate(axis = 0)
        else -> error("Cannot broadcast tensors")
    }
}

fun <T> NDArray<T>.broadcast(newShape: IntArray, asMatrixStack: Boolean = false): NDArray<T> {
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
    val preResult = this.reshape(newDims)

    return preResult.innerBroadcast(newShape, asMatrixStack)
}


fun <T> NDArray<T>.applyWithBroadcast(other: NDArray<T>, op: PrimitiveCombineFunction<T>): NDArray<T> {
    val newShape = broadcastShape(this.shape, other.shape)
    val castedThis = this.broadcast(newShape).array
    val castedOther = other.broadcast(newShape).array

    return NDArray(op.apply(castedThis, castedOther), type, newShape)
}
