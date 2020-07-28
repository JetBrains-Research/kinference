package org.jetbrains.research.kotlin.inference.data.tensors

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.concatenate
import kotlin.math.max

fun broadcastShape(currentShape: IntArray, newShape: IntArray): IntArray {
    val totalShapeLength = max(currentShape.size, newShape.size)
    val revCurrentShape = currentShape.reversedArray()
    val revNewShape = newShape.reversedArray()

    return IntArray(totalShapeLength) { i ->
        val currentDim = revCurrentShape.getOrNull(i) ?: 1
        val newDim = revNewShape.getOrNull(i) ?: 1

        if (currentDim != newDim && currentDim != 1 && newDim != 1) error("Cannot broadcast shapes")

        max(currentDim, newDim)
    }.reversedArray()
}

fun broadcastShape(currentShape: List<Int>, newShape: List<Int>): IntArray {
    return broadcastShape(currentShape.toIntArray(), newShape.toIntArray())
}

fun broadcastMatrixElementsShape(fstShape: IntArray, sndShape: IntArray): Pair<IntArray, IntArray> {
    val base = broadcastShape(fstShape.dropLast(2), sndShape.dropLast(2))

    val fst = base + fstShape.takeLast(2)
    val snd = base + sndShape.takeLast(2)
    return fst to snd
}

private fun NDArray.innerBroadcast(newShape: IntArray, asMatrixStack: Boolean = false): NDArray {
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

fun NDArray.broadcast(newShape: IntArray, asMatrixStack: Boolean = false): NDArray {
    if (this.shape.contentEquals(newShape)) return this

    val newDims = this.shape.copyOf().toMutableList()

    if (newShape.size > this.rank)
        repeat(newShape.size - this.rank) { newDims.add(0, 1) }

    val preResult = this.reshape(newDims.toIntArray())

    return preResult.innerBroadcast(newShape, asMatrixStack)
}


fun NDArray.applyWithBroadcast(other: NDArray, op: (Any, Any) -> Any): NDArray {
    val newShape = broadcastShape(this.shape, other.shape)
    val castedThis = this.broadcast(newShape).array
    val castedOther = other.broadcast(newShape).array

    return NDArray(op(castedThis, castedOther), type, shape)
}
