package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun transpose(array: FloatArray, rowNum: Int, colNum: Int): FloatArray {
    val result = FloatArray(rowNum * colNum)

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = array[i * colNum + j]
        }
    }

    return result
}

fun transpose(array: DoubleArray, rowNum: Int, colNum: Int): DoubleArray {
    val result = DoubleArray(rowNum * colNum)

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = array[i * colNum + j]
        }
    }

    return result
}

fun transpose(array: IntArray, rowNum: Int, colNum: Int): IntArray {
    val result = IntArray(rowNum * colNum)

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = array[i * colNum + j]
        }
    }

    return result
}

fun transpose(array: LongArray, rowNum: Int, colNum: Int): LongArray {
    val result = LongArray(rowNum * colNum)

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = array[i * colNum + j]
        }
    }

    return result
}

fun transpose(array: ShortArray, rowNum: Int, colNum: Int): ShortArray {
    val result = ShortArray(rowNum * colNum)

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = array[i * colNum + j]
        }
    }

    return result
}

fun NDArray.matrixTranspose(): NDArray {
    require(this.shape.size == 2)
    val newShape = shape.reversedArray()
    val newStrides = Strides(newShape)

    return when (array) {
        is IntArray -> IntNDArray(transpose(array, shape[0], shape[1]), newStrides)
        is FloatArray -> FloatNDArray(transpose(array, shape[0], shape[1]), newStrides)
        is ShortArray -> ShortNDArray(transpose(array, shape[0], shape[1]), newStrides)
        is DoubleArray -> DoubleNDArray(transpose(array, shape[0], shape[1]), newStrides)
        is LongArray -> LongNDArray(transpose(array, shape[0], shape[1]), newStrides)
        else -> throw UnsupportedOperationException()
    }
}
