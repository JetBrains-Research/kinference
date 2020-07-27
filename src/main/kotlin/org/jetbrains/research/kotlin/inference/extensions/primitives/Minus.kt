package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*

fun minus(left: FloatArray, right: FloatArray): FloatArray {
    return plus(left, -right)
}

fun minus(left: IntArray, right: IntArray): IntArray {
    return plus(left, -right)
}

fun minus(left: LongArray, right: LongArray): LongArray {
    return plus(left, -right)
}

fun minus(left: DoubleArray, right: DoubleArray): DoubleArray {
    return plus(left, -right)
}

fun minus(left: ShortArray, right: ShortArray): ShortArray {
    return plus(left, -right)
}

@Suppress("UNCHECKED_CAST")
inline fun NDArray.minus(x: Any): NDArray {
    return when (array) {
        is IntArray -> IntNDArray(minus(array, IntArray(this.linearSize) { x as Int }), strides)
        is FloatArray -> FloatNDArray(minus(array, FloatArray(this.linearSize) { x as Float }), strides)
        is ShortArray -> ShortNDArray(minus(array, ShortArray(this.linearSize) { x as Short }), strides)
        is DoubleArray -> DoubleNDArray(minus(array, DoubleArray(this.linearSize) { x as Double }), strides)
        is LongArray -> LongNDArray(minus(array, LongArray(this.linearSize) { x as Long }), strides)
        else -> throw UnsupportedOperationException()
    }
}
