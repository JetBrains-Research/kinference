package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.createArray

fun Collection<Long>.toIntArray(): IntArray = this.map { it.toInt() }.toIntArray()
fun Collection<Number>.toDoubleList(): List<Double> = this.map { it.toDouble() }
fun IntRange.reversed() = this.toList().reversed().toIntArray()

inline fun add(vararg terms: Number): Number = when (terms.first()) {
    is Float -> terms.reduce { acc, number -> acc.toFloat() + number.toFloat() }
    is Double -> terms.reduce { acc, number -> acc.toDouble() + number.toDouble() }
    is Int -> terms.reduce { acc, number -> acc.toInt() + number.toInt() }
    is Long -> terms.reduce { acc, number -> acc.toLong() + number.toLong() }
    else -> error("Unsupported data type")
}

inline fun times(vararg terms: Number): Number = when (terms.first()) {
    is Float -> terms.reduce { acc, number -> acc.toFloat() * number.toFloat() }
    is Double -> terms.reduce { acc, number -> acc.toDouble() * number.toDouble() }
    is Int -> terms.reduce { acc, number -> acc.toInt() * number.toInt() }
    is Long -> terms.reduce { acc, number -> acc.toLong() * number.toLong() }
    else -> error("Unsupported data type")
}

inline fun <reified T> NDArray<T>.max(): Number? {
    return when (array) {
        is IntArray -> array.max()
        is FloatArray -> array.max()
        is ShortArray -> array.max()
        is DoubleArray -> array.max()
        is LongArray -> array.max()
        else -> throw UnsupportedOperationException()
    }
}

inline fun <reified T> NDArray<T>.sum(): Number {
    return when (array) {
        is IntArray -> array.sum()
        is FloatArray -> array.sum()
        is ShortArray -> array.sum()
        is DoubleArray -> array.sum()
        is LongArray -> array.sum()
        else -> throw UnsupportedOperationException()
    }
}

inline fun <reified T> NDArray<T>.exp(): NDArray<T> {
    return when (array) {
        is FloatArray -> FloatNDArray((array as FloatArray).apply { for (i in this.indices) this[i] = kotlin.math.exp(this[i]) }, strides)
        is DoubleArray -> DoubleNDArray((array as DoubleArray).apply { for (i in this.indices) this[i] = kotlin.math.exp(this[i]) }, strides)
        else -> throw UnsupportedOperationException()
    } as NDArray<T>
}

inline fun <reified T : Any> NDArray<T>.scalarOp(x: Any, noinline op: (T, T) -> T): NDArray<T> {
    return NDArray(op(array, createArray(type, this.linearSize) { x } as T), type, strides)
}

