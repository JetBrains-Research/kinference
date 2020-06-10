package org.jetbrains.research.kotlin.mpp.inference.tensors

import scientifik.kmath.linear.GenericMatrixContext
import scientifik.kmath.linear.MatrixContext
import scientifik.kmath.operations.*
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.NDStructure
import kotlin.reflect.KClass

fun Collection<Long>.toIntArray() = this.map { it.toInt() }.toIntArray()

fun IntRange.reversed() = this.toList().reversed().toIntArray()

@Suppress("UNCHECKED_CAST")
fun <T : Any> resolveMatrixContext(kclass: KClass<T>) = when (kclass) {
    Double::class -> MatrixContext.auto(RealField)
    Float::class -> MatrixContext.auto(FloatField)
    Long::class -> MatrixContext.auto(LongRing)
    Int::class -> MatrixContext.auto(IntRing)
    else -> error("Unsupported data type")
} as GenericMatrixContext<T, Ring<T>>

fun add(x: Number, y: Number): Number = when (x) {
    is Float -> x + y.toFloat()
    is Double -> x + y.toDouble()
    is Int -> x + y.toInt()
    is Long -> x + y.toLong()
    else -> error("Unsupported data type")
}

fun times(x: Number, y: Number): Number = when (x) {
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
