package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import scientifik.kmath.structures.*

fun Tensor.splitWithAxis(parts: Int, axis: Int = 0): List<Tensor> {
    require(axis in data.shape.indices) { "Index $axis out of shape bound: (0, ${data.dimension - 1}" }

    val elementsByIndex = data.shape[axis]
    val mainSplit = elementsByIndex.div(parts)
    val split = MutableList(parts) { mainSplit }

    val tail = elementsByIndex % parts
    if (tail != 0) split.add(tail)

    return splitWithAxis(split.toIntArray(), axis)
}

fun Tensor.wrapOneDim(): Tensor {
    val newStrides = TensorStrides(intArrayOf(1, *this.data.shape))
    val buffer = BufferNDStructure(newStrides, this.data.buffer)
    return Tensor(this.info.name, buffer, this.info.type)
}

//if axis not 0
private fun Tensor.mergeOnAxis(other: Tensor, axis: Int): Tensor {
    val dim = this.data.shape
    val rows = MutableList(dim[0]) { i -> this.row(i).concatenate(other.row(i), axis - 1) }
    var result = rows[0]

    if (dim[0] > 1) result = rows.map { row -> row.wrapOneDim() }.reduce { acc, tensor -> acc.concatenate(tensor) }
    if (dim[0] == 1 && axis > 0) result = result.wrapOneDim()

    return result
}

fun Tensor.concatenate(other: Tensor, axis: Int = 0): Tensor {
    val actualAxis = if (axis < 0) this.data.shape.size + axis else axis
    if (actualAxis != 0) return this.mergeOnAxis(other, actualAxis)

    val fstDim: IntArray = this.data.shape
    var sndDim: IntArray = other.data.shape
    if (fstDim.size > 1 && sndDim.size == 1) sndDim = intArrayOf(1, sndDim[0])

    val newShape: IntArray = if (fstDim.size == 1) {
        intArrayOf(fstDim[0] + sndDim[0])
    } else {
        fstDim.copyOf(fstDim.size).apply { set(0, fstDim[0] + sndDim[0]) }
    }

    val allElements = this.data.buffer.asSequence() + other.data.buffer.asSequence()
    val buffer = BufferNDStructure(TensorStrides(newShape), allElements.toList().asBuffer())
    return Tensor("out", buffer, this.info.type)
}

fun Collection<Tensor>.concatenate(axis: Int): Tensor {
    return this.reduce { acc, tensor -> acc.concatenate(tensor, axis) }
}

fun Tensor.as2DList(): List<Tensor> {
    if (this.data.dimension == 2) return listOf(this)
    if (this.data.dimension == 1) return listOf(this.wrapOneDim())

    return this.rows().map { it.as2DList() }.flatten()
}
