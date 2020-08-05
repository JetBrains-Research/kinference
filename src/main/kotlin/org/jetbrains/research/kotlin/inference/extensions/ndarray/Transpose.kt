package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun transposeRec(prevArray: FloatArray, newArray: FloatArray, prevStrides: Strides, newStrides: Strides, index: Int, prevOffset: Int, newOffset: Int, permutation: IntArray) {
    if (index != newStrides.shape.lastIndex) {
        for (i in 0 until newStrides.shape[index])
            transposeRec(prevArray, newArray, prevStrides, newStrides, index + 1, prevOffset + prevStrides.strides[permutation[index]] * i,
                newOffset + newStrides.strides[index] * i, permutation)
    } else {
        val temp = prevStrides.strides[permutation[index]]
        for (i in 0 until newStrides.shape[index]) {
            newArray[newOffset + i] = prevArray[prevOffset + i * temp]
        }
    }
}

fun transpose(array: FloatArray, strides: Strides, permutation: IntArray): FloatNDArray {
    val newArray = FloatArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)

    transposeRec(array, newArray, strides, newStrides, 0, 0, 0, permutation)

    return FloatNDArray(newArray, newStrides)
}

fun transpose(array: DoubleArray, strides: Strides, permutation: IntArray): DoubleNDArray {
    val newArray = DoubleArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return DoubleNDArray(array, newStrides)
}

fun transpose(array: IntArray, strides: Strides, permutation: IntArray): IntNDArray {
    val newArray = IntArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return IntNDArray(newArray, newStrides)
}

fun transpose(array: LongArray, strides: Strides, permutation: IntArray): LongNDArray {
    val newArray = LongArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return LongNDArray(newArray, newStrides)
}

fun transpose(array: ShortArray, strides: Strides, permutation: IntArray): ShortNDArray {
    val newArray = ShortArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return ShortNDArray(newArray, newStrides)
}

fun <T> NDArray<T>.transpose(permutation: IntArray): NDArray<T> {
    return when (array) {
        is IntArray -> transpose(array, strides, permutation)
        is FloatArray -> transpose(array, strides, permutation)
        is ShortArray -> transpose(array, strides, permutation)
        is DoubleArray -> transpose(array, strides, permutation)
        is LongArray -> transpose(array, strides, permutation)
        else -> throw UnsupportedOperationException()
    } as NDArray<T>
}
