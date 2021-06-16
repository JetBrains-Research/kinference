package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.primitives.types.DataType

fun NDArray.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

private fun computeGatherShape(shape: IntArray, axis: Int, indices: NDArray): IntArray {
    val newShape = IntArray(shape.size + indices.rank - 1)
    shape.copyInto(newShape, 0, 0, axis)
    indices.shape.copyInto(newShape, axis)
    shape.copyInto(newShape, axis + indices.rank, axis + 1)

    return newShape
}

fun createGatherDstArray(axis: Int, indices: NDArray, shape: IntArray, type: DataType): MutableNDArray {
    val newShape = computeGatherShape(shape, axis, indices)
    return allocateNDArray(type, newShape)
}

fun NDArray.gather(indices: NDArray, axis: Int = 0): NDArray {
    val actualAxis = this.indexAxis(axis)
    val dst = createGatherDstArray(actualAxis, indices, shape, type)

    return this.gather(indices, axis, dst)
}

fun NDArray.gather(indices: NDArray, axis: Int = 0, dst: MutableNDArray): NDArray {
    val gatherOutputShape = computeGatherShape(shape, axis, indices)

    require(dst.shape.contentEquals(gatherOutputShape)) { "Incorrect destination shape" }

    val actualAxis = this.indexAxis(axis)

    val block = computeBlockSize(fromDim = actualAxis + 1)
    val dataBatch = computeBlockSize(fromDim = actualAxis)
    val indicesSize = indices.strides.linearSize
    val gatheredBatch = indicesSize * block

    val numBlocks = computeBlockSize(toDim = actualAxis)

    when (indices.type) {
        DataType.LONG -> {
            indices as LongNDArray

            val pointer = indices.array.pointer()
            for (numBatch in 0 until numBlocks) {
                var index = 0
                pointer.forEach(indicesSize) {
                    val idx = (if (it < 0) it + this.shape[actualAxis] else it).toInt()

                    val srcOffset = numBatch * dataBatch + idx * block
                    val dstOffset = numBatch * gatheredBatch + index++ * block
                    dst.copyFrom(dstOffset, this, srcOffset, srcOffset + block)
                }

                pointer.linearIndex = 0
            }
        }
        DataType.INT -> {
            indices as IntNDArray

            val pointer = indices.array.pointer()
            for (numBatch in 0 until numBlocks) {
                var index = 0
                pointer.forEach(indicesSize) {
                    val idx = if (it < 0) it + this.shape[actualAxis] else it

                    val srcOffset = numBatch * dataBatch + idx * block
                    val dstOffset = numBatch * gatheredBatch + index++ * block
                    dst.copyFrom(dstOffset, this, srcOffset, srcOffset + block)
                }

                pointer.linearIndex = 0
            }
        }
        else -> throw IllegalStateException("Indices array must have Long or Int type")
    }

    return dst
}
