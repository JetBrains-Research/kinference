package org.jetbrains.research.kotlin.inference.extensions.functional

interface PrimitiveArrayValueCombineFunction<T, V> {
    //fun apply(array: T, value: V): T
}

fun interface PrimitiveArrayWithScalar<T, V> : PrimitiveArrayValueCombineFunction<T, V> {
    fun apply(array: T, offset: Int, value: V, destination: T, destinationOffset: Int, size: Int)
}

fun interface FloatArrayWithFloat : PrimitiveArrayWithScalar<FloatArray, Float> {
    override fun apply(array: FloatArray, offset: Int, value: Float, destination: FloatArray, destinationOffset: Int, size: Int)
}

fun interface DoubleArrayWithDouble : PrimitiveArrayWithScalar<DoubleArray, Double> {
    override fun apply(array: DoubleArray, offset: Int, value: Double, destination: DoubleArray, destinationOffset: Int, size: Int)
}

fun interface IntArrayWithInt : PrimitiveArrayWithScalar<IntArray, Int> {
    override fun apply(array: IntArray, offset: Int, value: Int, destination: IntArray, destinationOffset: Int, size: Int)
}

fun interface LongArrayWithLong : PrimitiveArrayWithScalar<LongArray, Long> {
    override fun apply(array: LongArray, offset: Int, value: Long, destination: LongArray, destinationOffset: Int, size: Int)
}

fun interface ShortArrayWithShort : PrimitiveArrayWithScalar<ShortArray, Short> {
    override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int)
}
