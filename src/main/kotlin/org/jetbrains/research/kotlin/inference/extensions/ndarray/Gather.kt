package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.LongNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

fun <T> NDArray<T>.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

fun <T> createGatherDstArray(axis: Int, indices: LongNDArray, shape: IntArray, type: TensorProto.DataType): NDArray<T> {
    val newShape = IntArray(shape.size + indices.rank - 1)
    shape.copyInto(newShape, 0, 0, axis)
    indices.shape.copyInto(newShape, axis)
    shape.copyInto(newShape, axis + indices.rank, axis + 1)
    val newStrides = Strides(newShape)
    return allocateNDArray(type, newStrides)
}

fun <T> NDArray<T>.gather(indices: NDArray<Any>, axis: Int = 0): NDArray<T> {
    val actualAxis = this.indexAxis(axis)
    val dst = createGatherDstArray<T>(actualAxis, indices as LongNDArray, shape, type)

    val block = computeBlockSize(fromDim = actualAxis + 1)
    val dataBatch = computeBlockSize(fromDim = actualAxis)
    val indicesSize = indices.strides.linearSize
    val gatheredBatch = indicesSize * block

    val numBlocks = computeBlockSize(toDim = actualAxis)

    val indicesArray = IntArray(indices.array.size) { i ->
        if (indices.array[i] < 0) (indices.array[i].toInt() + this.shape[actualAxis]) else indices.array[i].toInt()
    }

    repeat(numBlocks * indicesSize) { index ->
        val numBatch = index / indicesSize
        val i = index % indicesSize
        val idx = indicesArray[i]

        val srcOffset = numBatch * dataBatch + idx * block
        val dstOffset = numBatch * gatheredBatch + i * block

        dst.place(dstOffset, this.array, srcOffset, srcOffset + block)
    }
    return dst
}
