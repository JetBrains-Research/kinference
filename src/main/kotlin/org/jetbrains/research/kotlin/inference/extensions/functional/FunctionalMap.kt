package org.jetbrains.research.kotlin.inference.extensions.functional

interface PrimitiveArrayFunction

fun interface FloatArrayToFloatArray : PrimitiveArrayFunction {
    fun apply(array: FloatArray): FloatArray
}

fun interface DoubleArrayToDoubleArray : PrimitiveArrayFunction {
    fun apply(array: DoubleArray): DoubleArray
}

fun interface IntArrayToIntArray : PrimitiveArrayFunction {
    fun apply(array: IntArray): IntArray
}

fun interface LongArrayToLongArray : PrimitiveArrayFunction {
    fun apply(array: LongArray): LongArray
}

fun interface ShortArrayToShortArray : PrimitiveArrayFunction {
    fun apply(array: ShortArray): ShortArray
}
