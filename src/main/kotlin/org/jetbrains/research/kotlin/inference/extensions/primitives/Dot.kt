package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun dot(left: FloatArray, right: FloatArray, leftShape: IntArray, rightShape: IntArray): FloatArray {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = FloatArray(leftShape[0] * rightShape[1])

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                array[i * rightShape[1] + j] += left[i * leftShape[1] + k] * right[k * rightShape[1] + j]
            }
        }
    }

    return array
}

fun dot(left: DoubleArray, right: DoubleArray, leftShape: IntArray, rightShape: IntArray): DoubleArray {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = DoubleArray(leftShape[0] * rightShape[1])

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                array[i * rightShape[1] + j] += left[i * leftShape[1] + k] * right[k * rightShape[1] + j]
            }
        }
    }

    return array
}

fun dot(left: IntArray, right: IntArray, leftShape: IntArray, rightShape: IntArray): IntArray {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = IntArray(leftShape[0] * rightShape[1])

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                array[i * rightShape[1] + j] += left[i * leftShape[1] + k] * right[k * rightShape[1] + j]
            }
        }
    }

    return array
}

fun dot(left: LongArray, right: LongArray, leftShape: IntArray, rightShape: IntArray): LongArray {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = LongArray(leftShape[0] * rightShape[1])

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                array[i * rightShape[1] + j] += left[i * leftShape[1] + k] * right[k * rightShape[1] + j]
            }
        }
    }

    return array
}

fun dot(left: ShortArray, right: ShortArray, leftShape: IntArray, rightShape: IntArray): ShortArray {
    require(leftShape.size == 2 && rightShape.size == 2)
    require(leftShape[1] == rightShape[0])

    val array = ShortArray(leftShape[0] * rightShape[1])

    for (i in 0 until leftShape[0]) {
        for (j in 0 until rightShape[1]) {
            for (k in 0 until leftShape[1]) {
                array[i * rightShape[1] + j] = (array[i * rightShape[1] + j] + left[i * leftShape[1] + k] * right[k * rightShape[1] + j]).toShort()
            }
        }
    }

    return array
}

fun NDArray.matrixDot(other: NDArray): NDArray {
    require(this::class == other::class)
    require(shape.size == 2 && other.shape.size == 2)
    require(shape[1] == other.shape[0])

    val newStrides = Strides(intArrayOf(shape[0], other.shape[1]))

    return when (array) {
        is FloatArray -> FloatNDArray(dot(array, other.array as FloatArray, shape, other.shape), newStrides)
        is IntArray -> IntNDArray(dot(array, other.array as IntArray, shape, other.shape), newStrides)
        is DoubleArray -> DoubleNDArray(dot(array, other.array as DoubleArray, shape, other.shape), newStrides)
        is ShortArray -> ShortNDArray(dot(array, other.array as ShortArray, shape, other.shape), newStrides)
        is LongArray -> LongNDArray(dot(array, other.array as LongArray, shape, other.shape), newStrides)
        else -> throw UnsupportedOperationException()
    }
}
