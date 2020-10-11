package io.kinference.ndarray.extensions

import io.kinference.ndarray.MutableNDArray
import io.kinference.ndarray.NDArray
import io.kinference.ndarray.Strides
import io.kinference.primitives.types.DataType

fun NDArray.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

@ExperimentalUnsignedTypes
fun createGatherDstArray(axis: Int, indices: NDArray, shape: IntArray, type: DataType): MutableNDArray {
    val newShape = IntArray(shape.size + indices.rank - 1)
    shape.copyInto(newShape, 0, 0, axis)
    indices.shape.copyInto(newShape, axis)
    shape.copyInto(newShape, axis + indices.rank, axis + 1)
    val newStrides = Strides(newShape)
    return allocateNDArray(type, newStrides)
}

@ExperimentalUnsignedTypes
fun NDArray.gather(indices: NDArray, axis: Int = 0): NDArray {
    val actualAxis = this.indexAxis(axis)
    val dst = createGatherDstArray(actualAxis, indices, shape, type)

    val block = computeBlockSize(fromDim = actualAxis + 1)
    val dataBatch = computeBlockSize(fromDim = actualAxis)
    val indicesSize = indices.strides.linearSize
    val gatheredBatch = indicesSize * block

    val numBlocks = computeBlockSize(toDim = actualAxis)

    val indicesArray = IntArray(indices.linearSize) { i ->
        val idx = (indices[i] as Number).toInt()
        if (idx < 0) idx + this.shape[actualAxis] else idx
    }

    repeat(numBlocks * indicesSize) { index ->
        val numBatch = index / indicesSize
        val i = index % indicesSize
        val idx = indicesArray[i]

        val srcOffset = numBatch * dataBatch + idx * block
        val dstOffset = numBatch * gatheredBatch + i * block

        dst.copyFrom(dstOffset, this, srcOffset, srcOffset + block)
    }
    return dst
}
