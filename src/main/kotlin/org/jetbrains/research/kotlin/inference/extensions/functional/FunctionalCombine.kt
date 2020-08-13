package org.jetbrains.research.kotlin.inference.extensions.functional

fun interface PrimitiveCombineFunction<T> {
    fun apply(array: T, otherArray: T): T
}

fun interface FloatArrayWithFloatArray : PrimitiveCombineFunction<FloatArray> {
    override fun apply(array: FloatArray, otherArray: FloatArray): FloatArray
}

fun interface DoubleArrayWithDoubleArray : PrimitiveCombineFunction<DoubleArray> {
    override fun apply(array: DoubleArray, otherArray: DoubleArray): DoubleArray
}

fun interface IntArrayWithIntArray : PrimitiveCombineFunction<IntArray> {
    override fun apply(array: IntArray, otherArray: IntArray): IntArray
}

fun interface LongArrayWithLongArray : PrimitiveCombineFunction<LongArray> {
    override fun apply(array: LongArray, otherArray: LongArray): LongArray
}

fun interface ShortArrayWithShortArray : PrimitiveCombineFunction<ShortArray> {
    override fun apply(array: ShortArray, otherArray: ShortArray): ShortArray
}
