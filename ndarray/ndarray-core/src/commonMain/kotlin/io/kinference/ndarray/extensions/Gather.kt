package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.primitives.types.DataType

internal fun computeGatherShape(shape: IntArray, axis: Int, indices: NDArray): IntArray {
    val newShape = IntArray(shape.size + indices.rank - 1)
    shape.copyInto(newShape, 0, 0, axis)
    indices.shape.copyInto(newShape, axis)
    shape.copyInto(newShape, axis + indices.rank, axis + 1)

    return newShape
}

internal suspend fun createGatherDstArray(axis: Int, indices: NDArray, shape: IntArray, type: DataType): MutableNDArrayCore {
    val newShape = computeGatherShape(shape, axis, indices)
    return allocateNDArray(type, newShape)
}

suspend fun gather(array: NDArrayCore, indices: NDArrayCore, axis: Int = 0): NDArrayCore {
    val actualAxis = array.indexAxis(axis)
    val dst = createGatherDstArray(actualAxis, indices, array.shape, array.type)

    return gather(array, indices, axis, dst)
}

fun gather(array: NDArrayCore, indices: NDArrayCore, axis: Int = 0, dst: MutableNDArrayCore): NDArrayCore {
    val gatherOutputShape = computeGatherShape(array.shape, axis, indices)

    require(dst.shape.contentEquals(gatherOutputShape)) { "Incorrect destination shape" }

    val actualAxis = array.indexAxis(axis)

    val block = array.computeBlockSize(fromDim = actualAxis + 1)
    val dataBatch = array.computeBlockSize(fromDim = actualAxis)
    val indicesSize = indices.strides.linearSize
    val gatheredBatch = indicesSize * block

    val numBlocks = array.computeBlockSize(toDim = actualAxis)

    when (indices.type) {
        DataType.LONG -> {
            indices as LongNDArray

            val pointer = indices.array.pointer()
            for (numBatch in 0 until numBlocks) {
                var index = 0
                pointer.forEach(indicesSize) {
                    val idx = (if (it < 0) it + array.shape[actualAxis] else it).toInt()

                    val srcOffset = numBatch * dataBatch + idx * block
                    val dstOffset = numBatch * gatheredBatch + index++ * block
                    dst.copyFrom(dstOffset, array, srcOffset, srcOffset + block)
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
                    val idx = if (it < 0) it + array.shape[actualAxis] else it

                    val srcOffset = numBatch * dataBatch + idx * block
                    val dstOffset = numBatch * gatheredBatch + index++ * block
                    dst.copyFrom(dstOffset, array, srcOffset, srcOffset + block)
                }

                pointer.linearIndex = 0
            }
        }
        else -> throw IllegalStateException("Indices array must have Long or Int type")
    }

    return dst
}
