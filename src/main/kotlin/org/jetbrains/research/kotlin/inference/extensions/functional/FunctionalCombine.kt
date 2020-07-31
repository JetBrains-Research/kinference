package org.jetbrains.research.kotlin.inference.extensions.functional

interface PrimitiveCombineFunction<T> {
    fun apply(array: T, otherArray: T): T
}

@FunctionalInterface
interface FloatArrayWithFloatArray: PrimitiveCombineFunction<FloatArray> {
    override fun apply(array: FloatArray, otherArray: FloatArray): FloatArray
}

@FunctionalInterface
interface DoubleArrayWithDoubleArray: PrimitiveCombineFunction<DoubleArray> {
    override fun apply(array: DoubleArray, otherArray: DoubleArray): DoubleArray
}

@FunctionalInterface
interface IntArrayWithIntArray: PrimitiveCombineFunction<IntArray> {
    override fun apply(array: IntArray, otherArray: IntArray): IntArray
}

@FunctionalInterface
interface LongArrayWithLongArray: PrimitiveCombineFunction<LongArray> {
    override fun apply(array: LongArray, otherArray: LongArray): LongArray
}

@FunctionalInterface
interface ShortArrayWithShortArray: PrimitiveCombineFunction<ShortArray> {
    override fun apply(array: ShortArray, otherArray: ShortArray): ShortArray
}
