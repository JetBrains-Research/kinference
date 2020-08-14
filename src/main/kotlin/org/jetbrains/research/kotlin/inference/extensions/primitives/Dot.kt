package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun dotInto(left: FloatArray, right: FloatArray, leftShape: IntArray, rightShape: IntArray, destination: FloatArray, clean: Boolean): FloatArray {
    if (clean) destination.fill(0.0f)

    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[ind + j] += temp * right[ind3 + j]
            }
        }
    }
    return destination
}

fun dotInto(left: DoubleArray, right: DoubleArray, leftShape: IntArray, rightShape: IntArray, destination: DoubleArray, clean: Boolean): DoubleArray {
    if (clean) destination.fill(0.0)

    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[ind + j] += temp * right[ind3 + j]
            }
        }
    }
    return destination
}

fun dotInto(left: IntArray, right: IntArray, leftShape: IntArray, rightShape: IntArray, destination: IntArray, clean: Boolean): IntArray {
    if (clean) destination.fill(0)

    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[ind + j] += temp * right[ind3 + j]
            }
        }
    }
    return destination
}

fun dotInto(left: LongArray, right: LongArray, leftShape: IntArray, rightShape: IntArray, destination: LongArray, clean: Boolean): LongArray {
    if (clean) destination.fill(0)

    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[ind + j] += temp * right[ind3 + j]
            }
        }
    }
    return destination
}

fun dotInto(left: ShortArray, right: ShortArray, leftShape: IntArray, rightShape: IntArray, destination: ShortArray, clean: Boolean): ShortArray {
    if (clean) destination.fill(0)

    val n = leftShape[0]
    val m = rightShape[1]
    val t = leftShape[1]

    for (i in 0 until n) {
        val ind = i * m
        val ind2 = i * t
        for (k in 0 until t) {
            val temp = left[ind2 + k]
            val ind3 = k * m
            for (j in 0 until m) {
                destination[ind + j] = (destination[ind + j] + temp * right[ind3 + j]).toShort()
            }
        }
    }
    return destination
}

fun <T> TypedNDArray<T>.matrixDot(other: TypedNDArray<T>): TypedNDArray<T> {
    require(this::class == other::class)
    require(shape.size == 2 && other.shape.size == 2)
    require(shape[1] == other.shape[0])

    val newStrides = Strides(intArrayOf(shape[0], other.shape[1]))

    @Suppress("UNCHECKED_CAST")
    return when (array) {
        is FloatArray -> FloatNDArray(dotInto(array as FloatArray, other.array as FloatArray, shape, other.shape, FloatArray(newStrides.linearSize), false), newStrides)
        is IntArray -> IntNDArray(dotInto(array as IntArray, other.array as IntArray, shape, other.shape, IntArray(newStrides.linearSize), false), newStrides)
        is DoubleArray -> DoubleNDArray(dotInto(array as DoubleArray, other.array as DoubleArray, shape, other.shape, DoubleArray(newStrides.linearSize), false), newStrides)
        is ShortArray -> ShortNDArray(dotInto(array as ShortArray, other.array as ShortArray, shape, other.shape, ShortArray(newStrides.linearSize), false), newStrides)
        is LongArray -> LongNDArray(dotInto(array as LongArray, other.array as LongArray, shape, other.shape, LongArray(newStrides.linearSize), false), newStrides)
        else -> throw UnsupportedOperationException()
    } as TypedNDArray<T>
}

fun <T> TypedNDArray<T>.matrixDotInto(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>, clean: Boolean = true): TypedNDArray<T> {
//    require(this::class == other::class)
    require(shape.size == 2 && other.shape.size == 2)
    require(shape[1] == other.shape[0])
    require(destination.shape[0] == shape[0] && destination.shape[1] == other.shape[1])

    when (array) {
        is FloatArray -> dotInto(array as FloatArray, other.array as FloatArray, shape, other.shape, destination.array as FloatArray, clean)
        is IntArray -> dotInto(array as IntArray, other.array as IntArray, shape, other.shape, destination.array as IntArray, clean)
        is DoubleArray -> dotInto(array as DoubleArray, other.array as DoubleArray, shape, other.shape, destination.array as DoubleArray, clean)
        is ShortArray -> dotInto(array as ShortArray, other.array as ShortArray, shape, other.shape, destination.array as ShortArray, clean)
        is LongArray -> dotInto(array as LongArray, other.array as LongArray, shape, other.shape, destination.array as LongArray, clean)
        else -> throw UnsupportedOperationException()
    }

    return destination
}
