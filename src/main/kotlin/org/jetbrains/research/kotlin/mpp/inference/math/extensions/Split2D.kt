package org.jetbrains.research.kotlin.mpp.inference.math.extensions

import org.jetbrains.research.kotlin.mpp.inference.data.tensors.TensorStrides
import scientifik.kmath.structures.*


fun splitHorizontal(buffer: FloatBuffer, parts: Int, rowNum: Int, colNum: Int): Array<NDBuffer<Float>> {
    require(colNum % parts == 0)

    val newCol = colNum / parts
    val arr = buffer.array

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = TensorStrides(newShape)

    return Array(parts) { num ->
        val result = FloatArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = arr[i * colNum + j + offset]
            }
        }
        BufferNDStructure(newStrides, result.asBuffer())
    }
}

fun splitHorizontal(buffer: DoubleBuffer, parts: Int, rowNum: Int, colNum: Int): Array<NDBuffer<Double>> {
    require(colNum % parts == 0)

    val newCol = colNum / parts
    val arr = buffer.array

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = TensorStrides(newShape)

    return Array(parts) { num ->
        val result = DoubleArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = arr[i * colNum + j + offset]
            }
        }
        BufferNDStructure(newStrides, result.asBuffer())
    }
}

fun splitHorizontal(buffer: IntBuffer, parts: Int, rowNum: Int, colNum: Int): Array<NDBuffer<Int>> {
    require(colNum % parts == 0)

    val newCol = colNum / parts
    val arr = buffer.array

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = TensorStrides(newShape)

    return Array(parts) { num ->
        val result = IntArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = arr[i * colNum + j + offset]
            }
        }
        BufferNDStructure(newStrides, result.asBuffer())
    }
}

fun splitHorizontal(buffer: LongBuffer, parts: Int, rowNum: Int, colNum: Int): Array<NDBuffer<Long>> {
    require(colNum % parts == 0)

    val newCol = colNum / parts
    val arr = buffer.array

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = TensorStrides(newShape)

    return Array(parts) { num ->
        val result = LongArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = arr[i * colNum + j + offset]
            }
        }
        BufferNDStructure(newStrides, result.asBuffer())
    }
}

fun splitHorizontal(buffer: ShortBuffer, parts: Int, rowNum: Int, colNum: Int): Array<NDBuffer<Short>> {
    require(colNum % parts == 0)

    val newCol = colNum / parts
    val arr = buffer.array

    val newShape = intArrayOf(rowNum, newCol)
    val newStrides = TensorStrides(newShape)

    return Array(parts) { num ->
        val result = ShortArray(newCol * rowNum)
        val offset = num * newCol
        for (i in (0 until rowNum)) {
            for (j in (0 until newCol)) {
                result[i * newCol + j] = arr[i * colNum + j + offset]
            }
        }
        BufferNDStructure(newStrides, result.asBuffer())
    }
}

fun <T : Any> NDBuffer<T>.splitHorizontal(parts: Int): Array<NDBuffer<T>> {
    require(shape.size == 2)
    return when (buffer) {
        is IntBuffer -> splitHorizontal(buffer as IntBuffer, parts, shape[0], shape[1])
        is FloatBuffer -> splitHorizontal(buffer as FloatBuffer, parts, shape[0], shape[1])
        is ShortBuffer -> splitHorizontal(buffer as ShortBuffer, parts, shape[0], shape[1])
        is DoubleBuffer -> splitHorizontal(buffer as DoubleBuffer, parts, shape[0], shape[1])
        is LongBuffer -> splitHorizontal(buffer as LongBuffer, parts, shape[0], shape[1])
        else -> throw UnsupportedOperationException()
    } as Array<NDBuffer<T>>
}
