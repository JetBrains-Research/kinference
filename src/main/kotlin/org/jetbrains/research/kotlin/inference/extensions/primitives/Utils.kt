package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.DoubleNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.FloatNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayCombineFunction
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

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

fun <T> NDArray<T>.max(): Number? {
    return when (array) {
        is IntArray -> array.max()
        is FloatArray -> array.max()
        is ShortArray -> array.max()
        is DoubleArray -> array.max()
        is LongArray -> array.max()
        else -> throw UnsupportedOperationException()
    }
}

fun <T> NDArray<T>.sum(): Number {
    return when (array) {
        is IntArray -> array.sum()
        is FloatArray -> array.sum()
        is ShortArray -> array.sum()
        is DoubleArray -> array.sum()
        is LongArray -> array.sum()
        else -> throw UnsupportedOperationException()
    }
}

fun <T> NDArray<T>.exp(): NDArray<T> {
    return when (array) {
        is FloatArray -> FloatNDArray((array as FloatArray).apply { for (i in this.indices) this[i] = kotlin.math.exp(this[i]) }, strides)
        is DoubleArray -> DoubleNDArray((array as DoubleArray).apply { for (i in this.indices) this[i] = kotlin.math.exp(this[i]) }, strides)
        else -> throw UnsupportedOperationException()
    } as NDArray<T>
}

fun <T : Any> NDArray<T>.scalarOp(x: Any, op: PrimitiveArrayCombineFunction<T>): NDArray<T> {
    val other = when (type) {
        TensorProto.DataType.DOUBLE -> DoubleArray(linearSize) { x as Double }
        TensorProto.DataType.FLOAT -> FloatArray(linearSize) { x as Float }
        TensorProto.DataType.INT64 -> LongArray(linearSize) { x as Long }
        TensorProto.DataType.INT32 -> IntArray(linearSize) { x as Int }
        TensorProto.DataType.INT16 -> ShortArray(linearSize) { x as Short }
        TensorProto.DataType.BOOL -> BooleanArray(linearSize) { x as Boolean }
        else -> error("Unsupported operator")
    }

    return NDArray(op.apply(array, other as T), type, strides)
}

