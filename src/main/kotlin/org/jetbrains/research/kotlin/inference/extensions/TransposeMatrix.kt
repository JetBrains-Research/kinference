package org.jetbrains.research.kotlin.inference.extensions

import org.jetbrains.research.kotlin.inference.data.tensors.TensorStrides
import scientifik.kmath.structures.*

fun transpose(buffer: FloatBuffer, rowNum: Int, colNum: Int): FloatBuffer {
    val result = FloatArray(rowNum * colNum)

    val arr = buffer.array

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = arr[i * colNum + j]
        }
    }

    return result.asBuffer()
}

fun transpose(buffer: DoubleBuffer, rowNum: Int, colNum: Int): DoubleBuffer {
    val result = DoubleArray(rowNum * colNum)

    val arr = buffer.array

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = arr[i * colNum + j]
        }
    }

    return result.asBuffer()
}

fun transpose(buffer: IntBuffer, rowNum: Int, colNum: Int): IntBuffer {
    val result = IntArray(rowNum * colNum)

    val arr = buffer.array

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = arr[i * colNum + j]
        }
    }

    return result.asBuffer()
}

fun transpose(buffer: LongBuffer, rowNum: Int, colNum: Int): LongBuffer {
    val result = LongArray(rowNum * colNum)

    val arr = buffer.array

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = arr[i * colNum + j]
        }
    }

    return result.asBuffer()
}

fun transpose(buffer: ShortBuffer, rowNum: Int, colNum: Int): ShortBuffer {
    val result = ShortArray(rowNum * colNum)

    val arr = buffer.array

    for (i in (0 until rowNum)) {
        for (j in (0 until colNum)) {
            result[j * rowNum + i] = arr[i * colNum + j]
        }
    }

    return result.asBuffer()
}

fun <T : Any> NDBuffer<T>.matrixTranspose(): NDBuffer<T> {
    require(this.shape.size == 2)
    val newShape = shape.reversedArray()
    val newStrides = TensorStrides(newShape)

    return when (buffer) {
        is IntBuffer -> BufferNDStructure(newStrides, transpose(this.buffer as IntBuffer, shape[0], shape[1]))
        is FloatBuffer -> BufferNDStructure(newStrides, transpose(this.buffer as FloatBuffer, shape[0], shape[1]))
        is ShortBuffer -> BufferNDStructure(newStrides, transpose(this.buffer as ShortBuffer, shape[0], shape[1]))
        is DoubleBuffer -> BufferNDStructure(newStrides, transpose(this.buffer as DoubleBuffer, shape[0], shape[1]))
        is LongBuffer -> BufferNDStructure(newStrides, transpose(this.buffer as LongBuffer, shape[0], shape[1]))
        else -> throw UnsupportedOperationException()
    } as BufferNDStructure<T>
}
