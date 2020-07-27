package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*

fun div(left: FloatArray, right: FloatArray): FloatArray {
    require(left.size == right.size)
    val array = FloatArray(left.size)

    for (i in left.indices) array[i] = left[i] / right[i]

    return array
}

fun div(left: IntArray, right: IntArray): IntArray {
    require(left.size == right.size)
    val array = IntArray(left.size)

    for (i in left.indices) array[i] = left[i] / right[i]

    return array
}

fun div(left: LongArray, right: LongArray): LongArray {
    require(left.size == right.size)
    val array = LongArray(left.size)

    for (i in left.indices) array[i] = left[i] / right[i]

    return array
}

fun div(left: DoubleArray, right: DoubleArray): DoubleArray {
    require(left.size == right.size)
    val array = DoubleArray(left.size)

    for (i in left.indices) array[i] = left[i] / right[i]

    return array
}

fun div(left: ShortArray, right: ShortArray): ShortArray {
    require(left.size == right.size)
    val array = ShortArray(left.size)

    for (i in left.indices) array[i] = (left[i] / right[i]).toShort()

    return array
}

fun NDArray.div(other: NDArray): NDArray {
    require(this::class == other::class)
    require(this.shape.contentEquals(other.shape))
    return when (array) {
        is IntArray -> IntNDArray(div(array, other.array as IntArray), strides)
        is FloatArray -> FloatNDArray(div(array, other.array as FloatArray), strides)
        is ShortArray -> ShortNDArray(div(array, other.array as ShortArray), strides)
        is DoubleArray -> DoubleNDArray(div(array, other.array as DoubleArray), strides)
        is LongArray -> LongNDArray(div(array, other.array as LongArray), strides)
        else -> throw UnsupportedOperationException()
    }
}

@Suppress("UNCHECKED_CAST")
inline fun NDArray.div(x: Any?): NDArray {
    return when (array) {
        is IntArray -> IntNDArray(div(array, IntArray(this.linearSize) { x as Int }), strides)
        is FloatArray -> FloatNDArray(div(array, FloatArray(this.linearSize) { x as Float }), strides)
        is ShortArray -> ShortNDArray(div(array, ShortArray(this.linearSize) { x as Short }), strides)
        is DoubleArray -> DoubleNDArray(div(array, DoubleArray(this.linearSize) { x as Double }), strides)
        is LongArray -> LongNDArray(div(array, LongArray(this.linearSize) { x as Long }), strides)
        else -> throw UnsupportedOperationException()
    }
}
