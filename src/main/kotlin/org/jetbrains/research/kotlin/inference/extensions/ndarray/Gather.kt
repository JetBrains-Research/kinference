package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.LongNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

private fun NDArray<Any>.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

private fun createGatherDstArray(axis: Int, indices: LongNDArray, shape: IntArray, type: TensorProto.DataType): NDArray<Any> {
    val addedShape = shape.toMutableList().also { it.removeAt(axis) }
    val newShape = addedShape.toMutableList().also { it.addAll(axis, indices.shape.toList()) }
    val newStrides = Strides(newShape.toIntArray())
    return allocateNDArray(type, newStrides)
}

fun NDArray<Any>.gather(indices: NDArray<Any>, axis: Int = 0): NDArray<Any> {
    val actualAxis = this.indexAxis(axis)
    val dst = createGatherDstArray(actualAxis, indices as LongNDArray, shape, type)

    val block = computeBlockSize(fromDim = actualAxis + 1)
    val dataBatch = computeBlockSize(fromDim = actualAxis)
    val indicesSize = indices.strides.linearSize
    val gatheredBatch = indicesSize * block

    val numBlocks = computeBlockSize(toDim = actualAxis)
    val indicesArray = indices.array.map { if (it < 0) (it.toInt() + this.shape[actualAxis]) else it.toInt()}.toIntArray()

    repeat(numBlocks * indicesSize) { index ->
        val numBatch = index / indicesSize
        val i = index % indicesSize
        val idx = indicesArray[i]

        val srcOffset = numBatch * dataBatch + idx * block
        val dstOffset = numBatch * gatheredBatch + i * block

        val slice = this.slice(block, srcOffset)
        dst.placeAll(dstOffset, slice)
    }
    return dst
}
