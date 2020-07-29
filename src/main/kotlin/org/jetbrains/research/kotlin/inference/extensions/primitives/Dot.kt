package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun dotInto(left: FloatArray, right: FloatArray, leftShape: IntArray, rightShape: IntArray, destination: FloatArray, clean: Boolean): FloatArray {
    if (clean) for (i in destination.indices) destination[i] = 0.0f

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                destination[i * rightShape[1] + j] += left[i * leftShape[1] + k] * right[k * rightShape[1] + j]
            }
        }
    }

    return destination
}

fun dotInto(left: DoubleArray, right: DoubleArray, leftShape: IntArray, rightShape: IntArray, destination: DoubleArray, clean: Boolean): DoubleArray {
    if (clean) for (i in destination.indices) destination[i] = 0.0

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                destination[i * rightShape[1] + j] += left[i * leftShape[1] + k] * right[k * rightShape[1] + j]
            }
        }
    }

    return destination
}

fun dotInto(left: IntArray, right: IntArray, leftShape: IntArray, rightShape: IntArray, destination: IntArray, clean: Boolean): IntArray {
    if (clean) for (i in destination.indices) destination[i] = 0

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                destination[i * rightShape[1] + j] += left[i * leftShape[1] + k] * right[k * rightShape[1] + j]
            }
        }
    }

    return destination
}

fun dotInto(left: LongArray, right: LongArray, leftShape: IntArray, rightShape: IntArray, destination: LongArray, clean: Boolean): LongArray {
    if (clean) for (i in destination.indices) destination[i] = 0

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                destination[i * rightShape[1] + j] += left[i * leftShape[1] + k] * right[k * rightShape[1] + j]
            }
        }
    }

    return destination
}

fun dotInto(left: ShortArray, right: ShortArray, leftShape: IntArray, rightShape: IntArray, destination: ShortArray, clean: Boolean): ShortArray {
    if (clean) for (i in destination.indices) destination[i] = 0

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                destination[i * rightShape[1] + j] = (destination[i * rightShape[1] + j] + left[i * leftShape[1] + k] * right[k * rightShape[1] + j]).toShort()
            }
        }
    }

    return destination
}

fun <T> NDArray<T>.matrixDot(other: NDArray<T>): NDArray<T> {
    require(this::class == other::class)
    require(shape.size == 2 && other.shape.size == 2)
    require(shape[1] == other.shape[0])

    val newStrides = Strides(intArrayOf(shape[0], other.shape[1]))

    @Suppress("UNCHECKED_CAST")
    return when (array) {
        is FloatArray -> FloatNDArray(dotInto(array, other.array as FloatArray, shape, other.shape, FloatArray(newStrides.linearSize), false), newStrides)
        is IntArray -> IntNDArray(dotInto(array, other.array as IntArray, shape, other.shape, IntArray(newStrides.linearSize), false), newStrides)
        is DoubleArray -> DoubleNDArray(dotInto(array, other.array as DoubleArray, shape, other.shape, DoubleArray(newStrides.linearSize), false), newStrides)
        is ShortArray -> ShortNDArray(dotInto(array, other.array as ShortArray, shape, other.shape, ShortArray(newStrides.linearSize), false), newStrides)
        is LongArray -> LongNDArray(dotInto(array, other.array as LongArray, shape, other.shape, LongArray(newStrides.linearSize), false), newStrides)
        else -> throw UnsupportedOperationException()
    } as NDArray<T>
}

fun <T> NDArray<T>.matrixDotInto(other: NDArray<T>, destination: NDArray<T>, clean: Boolean = true): NDArray<T> {
    require(this::class == other::class)
    require(shape.size == 2 && other.shape.size == 2)
    require(shape[1] == other.shape[0])
    require(destination.shape[0] == shape[0] && destination.shape[1] == other.shape[1])

    when (array) {
        is FloatArray -> dotInto(array, other.array as FloatArray, shape, other.shape, destination.array as FloatArray, clean)
        is IntArray -> dotInto(array, other.array as IntArray, shape, other.shape, destination.array as IntArray, clean)
        is DoubleArray -> dotInto(array, other.array as DoubleArray, shape, other.shape, destination.array as DoubleArray, clean)
        is ShortArray -> dotInto(array, other.array as ShortArray, shape, other.shape, destination.array as ShortArray, clean)
        is LongArray -> dotInto(array, other.array as LongArray, shape, other.shape, destination.array as LongArray, clean)
        else -> throw UnsupportedOperationException()
    }

    return destination
}
