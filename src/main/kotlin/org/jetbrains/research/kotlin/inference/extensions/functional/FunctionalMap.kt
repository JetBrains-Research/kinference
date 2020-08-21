package org.jetbrains.research.kotlin.inference.extensions.functional

interface PrimitiveArrayFunction

interface FloatArrayToFloatArray : PrimitiveArrayFunction {
    fun apply(array: FloatArray): FloatArray
}

interface DoubleArrayToDoubleArray : PrimitiveArrayFunction {
    fun apply(array: DoubleArray): DoubleArray
}

interface IntArrayToIntArray : PrimitiveArrayFunction {
    fun apply(array: IntArray): IntArray
}

interface LongArrayToLongArray : PrimitiveArrayFunction {
    fun apply(array: LongArray): LongArray
}

interface ShortArrayToShortArray : PrimitiveArrayFunction {
    fun apply(array: ShortArray): ShortArray
}
