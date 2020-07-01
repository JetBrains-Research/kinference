package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import scientifik.kmath.structures.BufferNDStructure
import kotlin.math.max

fun broadcastShape(currentShape: IntArray, newShape: IntArray): IntArray {
    val totalShapeLength = max(currentShape.size, newShape.size)
    val resultShape = MutableList(totalShapeLength) { -1 }
    val revCurrentShape = currentShape.reversed()
    val revNewShape = newShape.reversed()

    for (i in 0 until totalShapeLength) {
        val currentDim = revCurrentShape.getOrNull(i) ?: 1
        val newDim = revNewShape.getOrNull(i) ?: 1

        if (currentDim != newDim && currentDim != 1 && newDim != 1) error("Cannot broadcast shapes")

        resultShape[i] = max(currentDim, newDim)
    }

    return resultShape.reversed().toIntArray()
}

private fun Tensor.innerBroadcast(newShape: IntArray): Tensor {
    if (this.data.shape.contentEquals(newShape)) return this

    val castShape = newShape.copyOfRange(1, newShape.size)

    //broadcast is available only if corresponding dims are equal or at least one of them is 1
    return when (this.data.shape[0]) {
        1 -> {
            val rows = this.row(0).innerBroadcast(castShape)
            rows.reshape(intArrayOf(1, *castShape)).repeatRow(newShape[0])
        }
        newShape[0] -> this.rows().map { it.innerBroadcast(castShape) }.concatenate(axis = 0)
        else -> error("Cannot broadcast tensors")
    }
}

fun Tensor.broadcast(newShape: IntArray): Tensor {
    if (this.data.shape.contentEquals(newShape)) return this
    val newDims = this.data.shape.copyOf().toMutableList()

    if (newShape.size > this.data.dimension)
        repeat(newShape.size - this.data.dimension) { newDims.add(0, 1) }

    val preResult = this.reshape(newDims.toIntArray())

    return preResult.innerBroadcast(newShape)
}

fun Tensor.elementWiseWithBroadcast(other: Tensor, op: (Any, Any) -> Any): Tensor {
    val newShape = broadcastShape(this.data.shape, other.data.shape)
    val castedThis = this.broadcast(newShape).data as BufferNDStructure<Any>
    val castedOther = other.broadcast(newShape).data as BufferNDStructure<Any>

    val res = castedThis.ndCombine(castedOther) { fst, snd -> op(fst, snd) }
    return Tensor(this.info.name, res, this.info.type)
}
