package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun dotInto(left: FloatArray, leftOffset: Int, leftShape: IntArray,
            right: FloatArray, rightOffset: Int, rightShape: IntArray,
            destination: FloatArray, destinationOffset: Int): FloatArray {
    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[leftOffset + ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[destinationOffset + ind + j] += temp * right[rightOffset + ind3 + j]
            }
        }
    }
    return destination
}

fun dotInto(left: DoubleArray, leftOffset: Int, leftShape: IntArray,
            right: DoubleArray, rightOffset: Int, rightShape: IntArray,
            destination: DoubleArray, destinationOffset: Int): DoubleArray {
    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[leftOffset + ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[destinationOffset + ind + j] += temp * right[rightOffset + ind3 + j]
            }
        }
    }
    return destination
}

fun dotInto(left: IntArray, leftOffset: Int, leftShape: IntArray,
            right: IntArray, rightOffset: Int, rightShape: IntArray,
            destination: IntArray, destinationOffset: Int): IntArray {
    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[leftOffset + ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[destinationOffset + ind + j] += temp * right[rightOffset + ind3 + j]
            }
        }
    }
    return destination
}

fun dotInto(left: LongArray, leftOffset: Int, leftShape: IntArray,
            right: LongArray, rightOffset: Int, rightShape: IntArray,
            destination: LongArray, destinationOffset: Int): LongArray {
    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[leftOffset + ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[destinationOffset + ind + j] += temp * right[rightOffset + ind3 + j]
            }
        }
    }
    return destination
}

fun dotInto(left: ShortArray, leftOffset: Int, leftShape: IntArray,
            right: ShortArray, rightOffset: Int, rightShape: IntArray,
            destination: ShortArray, destinationOffset: Int): ShortArray {
    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[leftOffset + ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[destinationOffset + ind + j] = (destination[destinationOffset + ind + j] + temp * right[rightOffset + ind3 + j]).toShort()
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
        is FloatArray -> FloatNDArray(dotInto(array, offset, shape, other.array as FloatArray, other.offset, other.shape, FloatArray(newStrides.linearSize), 0), newStrides)
        is IntArray -> IntNDArray(dotInto(array, offset, shape, other.array as IntArray, other.offset, other.shape, IntArray(newStrides.linearSize), 0), newStrides)
        is DoubleArray -> DoubleNDArray(dotInto(array, offset, shape, other.array as DoubleArray, other.offset, other.shape, DoubleArray(newStrides.linearSize), 0), newStrides)
        is ShortArray -> ShortNDArray(dotInto(array, offset, shape, other.array as ShortArray, other.offset, other.shape, ShortArray(newStrides.linearSize), 0), newStrides)
        is LongArray -> LongNDArray(dotInto(array, offset, shape, other.array as LongArray, other.offset, other.shape, LongArray(newStrides.linearSize), 0), newStrides)
        else -> throw UnsupportedOperationException()
    } as NDArray<T>
}

fun <T> NDArray<T>.matrixDotInto(other: NDArray<T>, destination: NDArray<T>, clean: Boolean = true): NDArray<T> {
    if (clean) destination.clean()
//    require(this::class == other::class)
    require(shape.size == 2 && other.shape.size == 2)
    require(shape[1] == other.shape[0])
    require(destination.shape[0] == shape[0] && destination.shape[1] == other.shape[1])

    when (array) {
        is FloatArray -> dotInto(array, offset, shape, other.array as FloatArray, other.offset, other.shape, destination.array as FloatArray, destination.offset)
        is IntArray -> dotInto(array, offset, shape, other.array as IntArray, other.offset, other.shape, destination.array as IntArray, destination.offset) //dotInto(array, other.array as IntArray, shape, other.shape, destination.array as IntArray, clean)
        is DoubleArray -> dotInto(array, offset, shape, other.array as DoubleArray, other.offset, other.shape, destination.array as DoubleArray, destination.offset) //dotInto(array, other.array as DoubleArray, shape, other.shape, destination.array as DoubleArray, clean)
        is ShortArray -> dotInto(array, offset, shape, other.array as ShortArray, other.offset, other.shape, destination.array as ShortArray, destination.offset) //dotInto(array, other.array as ShortArray, shape, other.shape, destination.array as ShortArray, clean)
        is LongArray -> dotInto(array, offset, shape, other.array as LongArray, other.offset, other.shape, destination.array as LongArray, destination.offset) //dotInto(array, other.array as LongArray, shape, other.shape, destination.array as LongArray, clean)
        else -> throw UnsupportedOperationException()
    }

    return destination
}
