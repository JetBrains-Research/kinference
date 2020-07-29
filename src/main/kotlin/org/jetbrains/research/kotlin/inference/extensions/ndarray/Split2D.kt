package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun splitHorizontal(array: FloatArray, parts: Int, rowNum: Int, colNum: Int): Array<FloatNDArray> {
    require(colNum % parts == 0)

    val newCol = colNum / parts

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = Strides(newShape)

    return Array(parts) { num ->
        val result = FloatArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = array[i * colNum + j + offset]
            }
        }
        FloatNDArray(result, newStrides)
    }
}

fun splitHorizontal(array: DoubleArray, parts: Int, rowNum: Int, colNum: Int): Array<DoubleNDArray> {
    require(colNum % parts == 0)

    val newCol = colNum / parts

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = Strides(newShape)

    return Array(parts) { num ->
        val result = DoubleArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = array[i * colNum + j + offset]
            }
        }
        DoubleNDArray(result, newStrides)
    }
}

fun splitHorizontal(array: IntArray, parts: Int, rowNum: Int, colNum: Int): Array<IntNDArray> {
    require(colNum % parts == 0)

    val newCol = colNum / parts

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = Strides(newShape)

    return Array(parts) { num ->
        val result = IntArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = array[i * colNum + j + offset]
            }
        }
        IntNDArray(result, newStrides)
    }
}

fun splitHorizontal(array: LongArray, parts: Int, rowNum: Int, colNum: Int): Array<LongNDArray> {
    require(colNum % parts == 0)

    val newCol = colNum / parts

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = Strides(newShape)

    return Array(parts) { num ->
        val result = LongArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = array[i * colNum + j + offset]
            }
        }
        LongNDArray(result, newStrides)
    }
}

fun splitHorizontal(array: ShortArray, parts: Int, rowNum: Int, colNum: Int): Array<ShortNDArray> {
    require(colNum % parts == 0)

    val newCol = colNum / parts

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = Strides(newShape)

    return Array(parts) { num ->
        val result = ShortArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = array[i * colNum + j + offset]
            }
        }
        ShortNDArray(result, newStrides)
    }
}

@Suppress("UNCHECKED_CAST")
inline fun <reified T> NDArray<T>.splitHorizontal(parts: Int): Array<NDArray<T>> {
    require(shape.size == 2)
    return when (array) {
        is IntArray -> splitHorizontal(array, parts, shape[0], shape[1])
        is FloatArray -> splitHorizontal(array, parts, shape[0], shape[1])
        is ShortArray -> splitHorizontal(array, parts, shape[0], shape[1])
        is DoubleArray -> splitHorizontal(array, parts, shape[0], shape[1])
        is LongArray -> splitHorizontal(array, parts, shape[0], shape[1])
        else -> throw UnsupportedOperationException()
    } as Array<NDArray<T>>
}
