package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.primitives.toIntArray

fun NDArray.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<NDArray> {
    require(axis in shape.indices) { "Index $axis out of shape bound: (0, ${rank - 1}" }

    val elementsByIndex = shape[axis]
    val mainSplit = elementsByIndex / parts
    val split = MutableList(parts - 1) { mainSplit }

    val tail = elementsByIndex - mainSplit * (parts - 1)
    split.add(tail)

    return this.splitWithAxis(split.toIntArray(), axis, keepDims).toList()
}

fun NDArray.splitWithAxis(splitTensor: NDArray, axis: Int = 0, keepDims: Boolean = true): List<NDArray> {
    val split = splitTensor.array
    return if (splitTensor.linearSize == 1) {
        splitWithAxis((splitTensor[0] as Number).toInt(), axis, keepDims)
    } else {
        this.splitWithAxis((splitTensor.array as List<Long>).toIntArray(), axis, keepDims).toList()
    }
}

fun NDArray.wrapOneDim(): NDArray {
    val newStrides = Strides(intArrayOf(1, *this.shape))
    return this.clone(newStrides)
}

//if axis not 0
private fun NDArray.mergeOnAxis(other: NDArray, axis: Int): NDArray {
    val dim = this.shape
    val rows = this.rows.zip(other.rows).map { (fst, snd) -> fst.concatenate(snd, axis - 1) }
    var result = rows[0]

    if (dim[0] > 1) result = rows.map { row -> row.wrapOneDim() }.reduce { acc, tensor -> acc.concatenate(tensor) }
    if (dim[0] == 1 && axis > 0) result = result.wrapOneDim()

    return result
}

fun NDArray.concatenate(other: NDArray, axis: Int = 0): NDArray {
    val actualAxis = this.indexAxis(axis)
    if (actualAxis != 0) return this.mergeOnAxis(other, actualAxis)

    val fstDim: IntArray = this.shape
    var sndDim: IntArray = other.shape
    if (fstDim.size > 1 && sndDim.size == 1) sndDim = intArrayOf(1, sndDim[0])

    val newShape: IntArray = if (fstDim.size == 1) {
        intArrayOf(fstDim[0] + sndDim[0])
    } else {
        fstDim.copyOf(fstDim.size).apply { set(0, fstDim[0] + sndDim[0]) }
    }
    return allocateNDArray(type, Strides(newShape)).apply {
        placeAll(0, this@concatenate.array)
        placeAll(this@concatenate.linearSize, other.array)
    }
}

fun Collection<NDArray>.concatenate(axis: Int): NDArray {
    return this.reduce { acc, tensor -> acc.concatenate(tensor, axis) }
}

fun Array<NDArray>.stack(axis: Int): NDArray {
    return this.map {
        val newShape = this.first().shape.toMutableList()
        newShape.add(axis, 1)
        it.reshape(newShape.toIntArray())
    }.concatenate(axis)
}

fun NDArray.as2DList(): List<NDArray> {
    if (this.rank == 2) return listOf(this)
    if (this.rank == 1) return listOf(this.wrapOneDim())

    val matrixShape = intArrayOf(shape[indexAxis(-2)], shape[indexAxis(-1)])
    val matrixStrides = Strides(matrixShape)

    return List(strides.linearSize / matrixStrides.linearSize) { index ->
        val newData = createArray(type, matrixStrides.linearSize) {
            this[it + index * matrixStrides.linearSize]
        }
        NDArray(newData, type, matrixStrides.shape)
    }
    //return this.rows().map { it.as2DList() }.flatten()
}

fun NDArray.reshape(tensorShape: NDArray): NDArray {
    val requestedShape = tensorShape.array as LongArray
    val requestedShapeArray = IntArray(requestedShape.size) { i -> requestedShape[i].toInt() }
    require(requestedShapeArray.count { it == -1 } <= 1) { "At most one dimension of the new shape can be -1" }

    val newShape = requestedShapeArray.copyOf()
    for ((i, axisShape) in requestedShapeArray.withIndex()) {
        if (axisShape == 0) newShape[i] = shape[i]
    }

    val negativeIdx = newShape.indexOf(-1)
    if (negativeIdx != -1) {
        val elementsCount = newShape.filter { it != -1 }.reduce(Int::times)
        newShape[negativeIdx] = strides.linearSize / elementsCount
    }

    return reshape(newShape)
}
