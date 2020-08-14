package org.jetbrains.research.kotlin.inference.extensions.functional

fun interface PrimitiveArrayValueCombineFunction<T, V> {
    fun apply(array: T, value: V): T
}

fun interface FloatArrayWithFloat : PrimitiveArrayValueCombineFunction<FloatArray, Float> {
    override fun apply(array: FloatArray, value: Float): FloatArray
}

fun interface DoubleArrayWithDouble: PrimitiveArrayValueCombineFunction<DoubleArray, Double> {
    override fun apply(array: DoubleArray, value: Double): DoubleArray
}

fun interface IntArrayWithInt : PrimitiveArrayValueCombineFunction<IntArray, Int> {
    override fun apply(array: IntArray, value: Int): IntArray
}

fun interface LongArrayWithLong : PrimitiveArrayValueCombineFunction<LongArray, Long> {
    override fun apply(array: LongArray, value: Long): LongArray
}

fun interface ShortArrayWithShort : PrimitiveArrayValueCombineFunction<ShortArray, Short> {
    override fun apply(array: ShortArray, value: Short): ShortArray
}
