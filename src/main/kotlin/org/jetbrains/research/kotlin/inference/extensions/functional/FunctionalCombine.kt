package org.jetbrains.research.kotlin.inference.extensions.functional

fun interface PrimitiveArraysCombineFunction<T> : PrimitiveArrayValueCombineFunction<T, T> {
    override fun apply(array: T, value: T): T
}

fun interface FloatArrayWithFloatArray : PrimitiveArraysCombineFunction<FloatArray> {
    override fun apply(array: FloatArray, value: FloatArray): FloatArray
}

fun interface DoubleArrayWithDoubleArray : PrimitiveArraysCombineFunction<DoubleArray> {
    override fun apply(array: DoubleArray, value: DoubleArray): DoubleArray
}

fun interface IntArrayWithIntArray : PrimitiveArraysCombineFunction<IntArray> {
    override fun apply(array: IntArray, value: IntArray): IntArray
}

fun interface LongArrayWithLongArray : PrimitiveArraysCombineFunction<LongArray> {
    override fun apply(array: LongArray, value: LongArray): LongArray
}

fun interface ShortArrayWithShortArray : PrimitiveArraysCombineFunction<ShortArray> {
    override fun apply(array: ShortArray, value: ShortArray): ShortArray
}
