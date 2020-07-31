package org.jetbrains.research.kotlin.inference.extensions.functional

interface PrimitiveArrayFunction

@FunctionalInterface
interface FloatArrayToFloatArray: PrimitiveArrayFunction {
    fun apply(array: FloatArray): FloatArray
}

@FunctionalInterface
interface DoubleArrayToDoubleArray: PrimitiveArrayFunction {
    fun apply(array: DoubleArray): DoubleArray
}

@FunctionalInterface
interface IntArrayToIntArray: PrimitiveArrayFunction {
    fun apply(array: IntArray): IntArray
}

@FunctionalInterface
interface LongArrayToLongArray: PrimitiveArrayFunction {
    fun apply(array: LongArray): LongArray
}

@FunctionalInterface
interface ShortArrayToShortArray: PrimitiveArrayFunction {
    fun apply(array: ShortArray): ShortArray
}
