package org.jetbrains.research.kotlin.inference.extensions.functional

interface PrimitiveArraysCombineFunction<T> : PrimitiveArrayValueCombineFunction<T, T> {
    fun apply(left: T, leftOffset: Int, right: T, rightOffset: Int, destination: T, destinationOffset: Int, size: Int): T
}

interface FloatArrayWithFloatArray : PrimitiveArraysCombineFunction<FloatArray> {
    override fun apply(left: FloatArray, leftOffset: Int, right: FloatArray, rightOffset: Int, destination: FloatArray, destinationOffset: Int, size: Int): FloatArray
}

interface DoubleArrayWithDoubleArray : PrimitiveArraysCombineFunction<DoubleArray> {
    override fun apply(left: DoubleArray, leftOffset: Int, right: DoubleArray, rightOffset: Int, destination: DoubleArray, destinationOffset: Int, size: Int): DoubleArray
}

interface IntArrayWithIntArray : PrimitiveArraysCombineFunction<IntArray> {
    override fun apply(left: IntArray, leftOffset: Int, right: IntArray, rightOffset: Int, destination: IntArray, destinationOffset: Int, size: Int): IntArray
}

interface LongArrayWithLongArray : PrimitiveArraysCombineFunction<LongArray> {
    override fun apply(left: LongArray, leftOffset: Int, right: LongArray, rightOffset: Int, destination: LongArray, destinationOffset: Int, size: Int): LongArray
}

interface ShortArrayWithShortArray : PrimitiveArraysCombineFunction<ShortArray> {
    override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray
}
