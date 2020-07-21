package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import scientifik.kmath.linear.GenericMatrixContext
import scientifik.kmath.linear.MatrixContext
import scientifik.kmath.operations.*
import scientifik.kmath.structures.Buffer
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.NDStructure
import kotlin.reflect.KClass

fun Collection<Long>.toIntArray(): IntArray = this.map { it.toInt() }.toIntArray()
fun Collection<Number>.toDoubleList(): List<Double> = this.map { it.toDouble() }
fun Buffer<Number>.toIntArray() = IntArray(this.size) { i -> this[i].toInt() }
fun Buffer<Number>.toDoubleArray() = DoubleArray(this.size) { i -> this[i].toDouble() }

fun IntRange.reversed() = this.toList().reversed().toIntArray()

@Suppress("UNCHECKED_CAST")
fun <T : Any> resolveMatrixContext(kClass: KClass<T>) = when (kClass) {
    Double::class -> MatrixContext.auto(RealField)
    Float::class -> MatrixContext.auto(FloatField)
    Long::class -> MatrixContext.auto(LongRing)
    Int::class -> MatrixContext.auto(IntRing)
    else -> error("Unsupported data type")
} as GenericMatrixContext<T, Ring<T>>

inline fun add(vararg terms: Number): Number = when (terms.first()) {
    is Float -> terms.reduce { acc, number -> acc.toFloat() + number.toFloat() }
    is Double -> terms.reduce { acc, number -> acc.toDouble() + number.toDouble() }
    is Int -> terms.reduce { acc, number -> acc.toInt() + number.toInt() }
    is Long -> terms.reduce { acc, number -> acc.toLong() + number.toLong() }
    else -> error("Unsupported data type")
}

inline fun times(x: Number, y: Number): Number = when (x) {
    is Float -> x * y.toFloat()
    is Double -> x * y.toDouble()
    is Int -> x * y.toInt()
    is Long -> x * y.toLong()
    else -> error("Unsupported data type")
}

inline fun <reified T : Any> BufferNDStructure<T>.ndCombine(
    struct: BufferNDStructure<T>,
    crossinline block: (T, T) -> T
): BufferNDStructure<T> {
    if (!this.shape.contentEquals(struct.shape)) error("Shape mismatch in structure combination")
    return NDStructure.build(this.strides) { block(this[it], struct[it]) }
}
