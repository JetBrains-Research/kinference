package org.jetbrains.research.kotlin.mpp.inference.mathExtension

import org.jetbrains.research.kotlin.mpp.inference.data.tensors.TensorStrides
import scientifik.kmath.structures.*

fun dot(left: FloatBuffer, right: FloatBuffer, leftShape: IntArray, rightShape: IntArray) : FloatBuffer {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = FloatArray(leftShape[0] * rightShape[1])

    val a = left.array
    val b = right.array

    for (i in (0 until leftShape[0])) {
        for (j in (0 until rightShape[1])) {
            for (k in (0 until leftShape[1])) {
                array[i * rightShape[1] + j] += a[i * leftShape[1] + k] * b[k * rightShape[1] + j]
            }
        }
    }

    return array.asBuffer()
}

fun dot(left: DoubleBuffer, right: DoubleBuffer, leftShape: IntArray, rightShape: IntArray) : DoubleBuffer {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = DoubleArray(leftShape[0] * rightShape[1])

    val a = left.array
    val b = right.array

    for (i in (0 until leftShape[0])) {
        for (j in (0 until rightShape[1])) {
            for (k in (0 until leftShape[1])) {
                array[i * rightShape[1] + j] += a[i * leftShape[1] + k] * b[k * rightShape[1] + j]
            }
        }
    }

    return array.asBuffer()
}

fun dot(left: IntBuffer, right: IntBuffer, leftShape: IntArray, rightShape: IntArray) : IntBuffer {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = IntArray(leftShape[0] * rightShape[1])

    val a = left.array
    val b = right.array

    for (i in (0 until leftShape[0])) {
        for (j in (0 until rightShape[1])) {
            for (k in (0 until leftShape[1])) {
                array[i * rightShape[1] + j] += a[i * leftShape[1] + k] * b[k * rightShape[1] + j]
            }
        }
    }

    return array.asBuffer()
}

fun dot(left: LongBuffer, right: LongBuffer, leftShape: IntArray, rightShape: IntArray) : LongBuffer {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = LongArray(leftShape[0] * rightShape[1])

    val a = left.array
    val b = right.array

    for (i in (0 until leftShape[0])) {
        for (j in (0 until rightShape[1])) {
            for (k in (0 until leftShape[1])) {
                array[i * rightShape[1] + j] += a[i * leftShape[1] + k] * b[k * rightShape[1] + j]
            }
        }
    }

    return array.asBuffer()
}

fun dot(left: ShortBuffer, right: ShortBuffer, leftShape: IntArray, rightShape: IntArray) : ShortBuffer {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = ShortArray(leftShape[0] * rightShape[1])

    val a = left.array
    val b = right.array

    for (i in (0 until leftShape[0])) {
        for (j in (0 until rightShape[1])) {
            for (k in (0 until leftShape[1])) {
                array[i * rightShape[1] + j] = (array[i * rightShape[1] + j] + a[i * leftShape[1] + k] * b[k * rightShape[1] + j]).toShort()
            }
        }
    }

    return array.asBuffer()
}

fun <T : Any> NDBuffer<T>.dot(other: NDBuffer<T>): NDBuffer<T> {
    require(this::class == other::class)
    require(shape.size == 2 && other.shape.size == 2)
    require(shape[1] == other.shape[0])

    val newStrides = TensorStrides(intArrayOf(shape[0], other.shape[1]))

    return when(buffer) {
        is FloatBuffer -> BufferNDStructure(newStrides, dot(this.buffer as FloatBuffer, other.buffer as FloatBuffer, this.shape, other.shape))
        is IntBuffer -> BufferNDStructure(newStrides, dot(this.buffer as IntBuffer, other.buffer as IntBuffer, this.shape, other.shape))
        is DoubleBuffer -> BufferNDStructure(newStrides, dot(this.buffer as DoubleBuffer, other.buffer as DoubleBuffer, this.shape, other.shape))
        is ShortBuffer -> BufferNDStructure(newStrides, dot(this.buffer as ShortBuffer, other.buffer as ShortBuffer, this.shape, other.shape))
        is LongBuffer -> BufferNDStructure(newStrides, dot(this.buffer as LongBuffer, other.buffer as LongBuffer, this.shape, other.shape))
        else -> throw UnsupportedOperationException()
    } as BufferNDStructure<T>
}
