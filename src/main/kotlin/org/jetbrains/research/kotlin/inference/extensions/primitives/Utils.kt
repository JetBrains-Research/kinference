package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*

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

inline fun NDArray.max(): Any? {
    return when (array) {
        is IntArray -> array.max()
        is FloatArray -> array.max()
        is ShortArray -> array.max()
        is DoubleArray -> array.max()
        is LongArray -> array.max()
        else -> throw UnsupportedOperationException()
    }
}

fun NDArray.sum(): Any {
    return when (array) {
        is IntArray -> array.sum()
        is FloatArray -> array.sum()
        is ShortArray -> array.sum()
        is DoubleArray -> array.sum()
        is LongArray -> array.sum()
        else -> throw UnsupportedOperationException()
    }
}

inline fun NDArray.exp(): NDArray {
    return when (array) {
        is FloatArray -> FloatNDArray(array.apply { for (i in this.indices) this[i] = kotlin.math.exp(this[i]) }, strides)
        is DoubleArray -> DoubleNDArray(array.apply { for (i in this.indices) this[i] = kotlin.math.exp(this[i]) }, strides)
        else -> throw UnsupportedOperationException()
    }
}
