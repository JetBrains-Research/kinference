package org.jetbrains.research.kotlin.mpp.inference.tensors

import scientifik.kmath.structures.*

fun Tensor.as2DCollection(): Collection<Tensor> {
    require(data.dimension == 3)

    val blockSize = data.shape[1] * data.shape[2]
    val newShape = intArrayOf(data.shape[1], data.shape[2])
    val newStrides = TensorStrides(newShape)
    return List(data.shape[0]) { index ->
        val newBuffer = VirtualBuffer(blockSize) { i ->
            val indices = newStrides.index(i)
            val rowNum = indices[0]
            val colNum = indices[1]
            data[index, rowNum, colNum]
        }
        val newStructure = BufferNDStructure(newStrides, newBuffer)
        Tensor(null, newStructure, type)
    }
}


fun Tensor.splitWithAxis(parts: Int, axis: Int = 0): List<Tensor> {
    require(axis in data.shape.indices) { "Index $axis out of shape bound: (0, ${data.dimension - 1}" }

    val elementsByIndex = data.shape[axis]
    val mainSplit = elementsByIndex.div(parts)
    val split = MutableList(parts) { mainSplit }

    val tail = elementsByIndex % parts
    if (tail != 0) split.add(tail)

    return splitWithAxis(split.toIntArray(), axis)
}
