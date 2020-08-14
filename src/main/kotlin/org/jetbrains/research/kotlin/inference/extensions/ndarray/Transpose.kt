package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

private fun Strides.transpose(permutations: IntArray): Strides {
    val newShape = IntArray(shape.size)
    for ((i, axis) in permutations.withIndex()) {
        newShape[i] = shape[axis]
    }
    return Strides(newShape)
}

private fun Strides.permuteIndicesAt(i: Int, permutation: IntArray): IntArray {
    val indices = this.index(i)
    val newIndices = IntArray(indices.size)
    for ((id, axis) in permutation.withIndex()) {
        newIndices[axis] = indices[id]
    }
    return newIndices
}

fun transposeRec(prevArray: FloatArray, newArray: FloatArray, prevStrides: Strides, newStrides: Strides, index: Int, prevOffset: Int, newOffset: Int, permutation: IntArray) {
    if (index != newStrides.shape.lastIndex) {
        val temp = prevStrides.strides[permutation[index]]
        val temp2 = newStrides.strides[index]
        for (i in 0 until newStrides.shape[index])
            transposeRec(prevArray, newArray, prevStrides, newStrides, index + 1, prevOffset + temp * i,
                newOffset + temp2 * i, permutation)
    } else {
        val temp = prevStrides.strides[permutation[index]]
        if (temp == 1) {
            prevArray.copyInto(newArray, newOffset, prevOffset, prevOffset + newStrides.shape[index])
        } else {
            for (i in 0 until newStrides.shape[index]) {
                newArray[newOffset + i] = prevArray[prevOffset + i * temp]
            }
        }
    }
}

fun transpose(array: FloatArray, strides: Strides, newStrides: Strides, permutation: IntArray): FloatArray {
    transposeRec(array.copyOf(), array, strides, newStrides, 0, 0, 0, permutation)

    return array
}

fun transpose(array: DoubleArray, strides: Strides, newStrides: Strides, permutation: IntArray): DoubleArray {
    val tmp = array.copyOf()

    for (i in array.indices) {
        val newIndices = newStrides.permuteIndicesAt(i, permutation)
        array[i] = tmp[strides.offset(newIndices)]
    }

    return array
}

fun transpose(array: IntArray, strides: Strides, newStrides: Strides, permutation: IntArray): IntArray {
    val tmp = array.copyOf()

    for (i in array.indices) {
        val newIndices = newStrides.permuteIndicesAt(i, permutation)
        array[i] = tmp[strides.offset(newIndices)]
    }

    return array
}

fun transpose(array: LongArray, strides: Strides, newStrides: Strides, permutation: IntArray): LongArray {
    val tmp = array.copyOf()

    for (i in array.indices) {
        val newIndices = newStrides.permuteIndicesAt(i, permutation)
        array[i] = tmp[strides.offset(newIndices)]
    }

    return tmp
}

fun transpose(array: ShortArray, strides: Strides, newStrides: Strides, permutation: IntArray): ShortArray {
    val tmp = array.copyOf()

    for (i in array.indices) {
        val newIndices = newStrides.permuteIndicesAt(i, permutation)
        array[i] = tmp[strides.offset(newIndices)]
    }

    return array
}

fun <T> MutableTypedNDArray<T>.transpose(permutations: IntArray): MutableTypedNDArray<T> {
    val newStrides = strides.transpose(permutations)

    when (array) {
        is IntArray -> transpose(array as IntArray, strides, newStrides, permutations)
        is FloatArray -> transpose(array as FloatArray, strides, newStrides, permutations)
        is ShortArray -> transpose(array as ShortArray, strides, newStrides, permutations)
        is DoubleArray -> transpose(array as DoubleArray, strides, newStrides, permutations)
        is LongArray -> transpose(array as LongArray, strides, newStrides, permutations)
        else -> throw UnsupportedOperationException()
    }

    return this.reshape(newStrides)
}
