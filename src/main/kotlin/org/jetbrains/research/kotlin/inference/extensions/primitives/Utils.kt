package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.functional.*

fun Collection<Number>.toIntArray(): IntArray {
    val array = IntArray(this.size)
    for ((i, element) in this.withIndex()) {
        array[i] = element.toInt()
    }
    return array
}

fun IntRange.reversed(): IntArray {
    val size = this.last - this.first + 1
    val array = IntArray(size)
    for ((i, element) in this.withIndex()) {
        array[size - i - 1] = element
    }
    return array
}

fun IntRange.toIntArray(): IntArray {
    val size = this.last - this.first + 1
    val array = IntArray(size)
    for ((i, element) in this.withIndex()) {
        array[i] = element
    }
    return array
}

fun add(vararg terms: Number): Number = when (terms.first()) {
    is Float -> terms.reduce { acc, number -> acc.toFloat() + number.toFloat() }
    is Double -> terms.reduce { acc, number -> acc.toDouble() + number.toDouble() }
    is Int -> terms.reduce { acc, number -> acc.toInt() + number.toInt() }
    is Long -> terms.reduce { acc, number -> acc.toLong() + number.toLong() }
    else -> error("Unsupported data type")
}

fun times(vararg terms: Number): Number = when (terms.first()) {
    is Float -> terms.reduce { acc, number -> acc.toFloat() * number.toFloat() }
    is Double -> terms.reduce { acc, number -> acc.toDouble() * number.toDouble() }
    is Int -> terms.reduce { acc, number -> acc.toInt() * number.toInt() }
    is Long -> terms.reduce { acc, number -> acc.toLong() * number.toLong() }
    else -> error("Unsupported data type")
}

fun <T> TypedNDArray<T>.max(): Number? {
    return when (array) {
        is IntArray -> (array as IntArray).max()
        is FloatArray -> (array as FloatArray).max()
        is ShortArray -> (array as ShortArray).max()
        is DoubleArray -> (array as DoubleArray).max()
        is LongArray -> (array as LongArray).max()
        else -> throw UnsupportedOperationException()
    }
}

fun <T> TypedNDArray<T>.sum(): Number {
    return when (array) {
        is IntArray -> (array as IntArray).sum()
        is FloatArray -> (array as FloatArray).sum()
        is ShortArray -> (array as ShortArray).sum().toShort()
        is DoubleArray -> (array as DoubleArray).sum()
        is LongArray -> (array as LongArray).sum()
        else -> throw UnsupportedOperationException()
    }
}

fun <T> MutableTypedNDArray<T>.exp(): MutableTypedNDArray<T> {
    when (array) {
        is FloatArray -> mapElements(FloatArrayToFloatArray { array -> for (i in array.indices) array[i] = kotlin.math.exp(array[i]); array })
        is DoubleArray -> mapElements(FloatArrayToFloatArray { array -> for (i in array.indices) array[i] = kotlin.math.exp(array[i]); array })
        else -> throw UnsupportedOperationException()
    } as NDArray<T>
    return this
}

fun <T : Any, V : Any> TypedNDArray<T>.scalarOp(x: V, op: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    op.apply(array, x)
    return this
}

fun Int.concat(array: IntArray): IntArray {
    val copy = IntArray(array.size + 1)
    System.arraycopy(array, 0, copy, 1, array.size)
    copy[0] = this
    return copy
}

fun IntArray.concat(value: Int): IntArray {
    val copy = IntArray(size + 1)
    System.arraycopy(this, 0, copy, 0, size)
    copy[size] = value
    return copy
}
