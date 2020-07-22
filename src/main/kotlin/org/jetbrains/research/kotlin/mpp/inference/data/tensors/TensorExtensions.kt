package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import org.jetbrains.research.kotlin.mpp.inference.*
import scientifik.kmath.structures.*

fun Tensor.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
    require(axis in data.shape.indices) { "Index $axis out of shape bound: (0, ${data.dimension - 1}" }

    val elementsByIndex = data.shape[axis]
    val mainSplit = elementsByIndex / parts
    val split = MutableList(parts) { mainSplit }

    val tail = elementsByIndex % parts
    if (tail != 0) split.add(tail)

    return this.splitWithAxis(split.toIntArray(), axis, keepDims)
}

fun Tensor.splitWithAxis(splitTensor: Tensor, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
    val split = splitTensor.data.buffer as Buffer<Number>
    return if (split.size == 1) {
        splitWithAxis(split[0].toInt(), axis, keepDims)
    } else {
        this.splitWithAxis(split.toIntArray(), axis, keepDims)
    }
}

fun Tensor.wrapOneDim(): Tensor {
    val newStrides = TensorStrides(intArrayOf(1, *this.data.shape))
    val buffer = BufferNDStructure(newStrides, this.data.buffer)
    return Tensor(this.info.name, buffer, this.info.type)
}

//if axis not 0
private fun Tensor.mergeOnAxis(other: Tensor, axis: Int): Tensor {
    val dim = this.data.shape
    val rows = this.rows.zip(other.rows).map { (fst, snd) -> fst.concatenate(snd, axis - 1) }
    var result = rows[0]

    if (dim[0] > 1) result = rows.map { row -> row.wrapOneDim() }.reduce { acc, tensor -> acc.concatenate(tensor) }
    if (dim[0] == 1 && axis > 0) result = result.wrapOneDim()

    return result
}

fun Tensor.concatenate(other: Tensor, axis: Int = 0): Tensor {
    val actualAxis = this.indexAxis(axis)
    if (actualAxis != 0) return this.mergeOnAxis(other, actualAxis)

    val fstDim: IntArray = this.data.shape
    var sndDim: IntArray = other.data.shape
    if (fstDim.size > 1 && sndDim.size == 1) sndDim = intArrayOf(1, sndDim[0])

    val newShape: IntArray = if (fstDim.size == 1) {
        intArrayOf(fstDim[0] + sndDim[0])
    } else {
        fstDim.copyOf(fstDim.size).apply { set(0, fstDim[0] + sndDim[0]) }
    }

    val allElements = allocateMutableBuffer(this.info.type, this.data.buffer.size + other.data.buffer.size).apply {
        placeAll(data.buffer)
        placeAll(other.data.buffer, index = data.buffer.size)
    }

    val buffer = BufferNDStructure(TensorStrides(newShape), allElements)
    return Tensor("out", buffer, this.info.type)
}

fun Collection<Tensor>.concatenate(axis: Int): Tensor {
    return this.reduce { acc, tensor -> acc.concatenate(tensor, axis) }
}

fun Collection<Tensor>.stack(axis: Int): Tensor {
    return this.map {
        val newShape = this.first().data.shape.toMutableList()
        newShape.add(axis, 1)
        it.reshape(newShape.toIntArray())
    }.concatenate(axis)
}

fun Tensor.as2DList(): List<Tensor> {
    if (this.data.dimension == 2) return listOf(this)
    if (this.data.dimension == 1) return listOf(this.wrapOneDim())

    val matrixShape = intArrayOf(data.shape[indexAxis(-2)], data.shape[indexAxis(-1)])
    val matrixStrides = TensorStrides(matrixShape)
    val ans = List(data.strides.linearSize / matrixStrides.linearSize) { index ->
        val newBuffer = createBuffer(info.type, matrixStrides.linearSize) {
            (data.buffer[it + index * matrixStrides.linearSize] as Number).toFloat()
        } as Buffer<Any>
        val newStructure = BufferNDStructure(matrixStrides, newBuffer)
        Tensor(null, newStructure, info.type)
    }
    return ans
    //return this.rows().map { it.as2DList() }.flatten()
}

