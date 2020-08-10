package org.jetbrains.research.kotlin.inference.extensions.functional

fun interface PrimitiveArrayCombineFunction<T> {
    fun apply(array: T, otherArray: T): T
}

fun interface FloatArrayWithFloatArray : PrimitiveArrayCombineFunction<FloatArray> {
    override fun apply(array: FloatArray, otherArray: FloatArray): FloatArray
}

fun interface DoubleArrayWithDoubleArray : PrimitiveArrayCombineFunction<DoubleArray> {
    override fun apply(array: DoubleArray, otherArray: DoubleArray): DoubleArray
}

fun interface IntArrayWithIntArray : PrimitiveArrayCombineFunction<IntArray> {
    override fun apply(array: IntArray, otherArray: IntArray): IntArray
}

fun interface LongArrayWithLongArray : PrimitiveArrayCombineFunction<LongArray> {
    override fun apply(array: LongArray, otherArray: LongArray): LongArray
}

fun interface ShortArrayWithShortArray : PrimitiveArrayCombineFunction<ShortArray> {
    override fun apply(array: ShortArray, otherArray: ShortArray): ShortArray
}

//fun interface PrimitiveScalarCombineFunction<T> {
//    fun apply(array: T, otherArray: T): T
//}
//
//fun interface FloatArrayWithFloatArray : PrimitiveArrayCombineFunction<FloatArray> {
//    override fun apply(array: FloatArray, otherArray: FloatArray): FloatArray
//}
//
//fun interface DoubleArrayWithDoubleArray : PrimitiveArrayCombineFunction<DoubleArray> {
//    override fun apply(array: DoubleArray, otherArray: DoubleArray): DoubleArray
//}
//
//fun interface IntArrayWithIntArray : PrimitiveArrayCombineFunction<IntArray> {
//    override fun apply(array: IntArray, otherArray: IntArray): IntArray
//}
//
//fun interface LongArrayWithLongArray : PrimitiveArrayCombineFunction<LongArray> {
//    override fun apply(array: LongArray, otherArray: LongArray): LongArray
//}
//
//fun interface ShortArrayWithShortArray : PrimitiveArrayCombineFunction<ShortArray> {
//    override fun apply(array: ShortArray, otherArray: ShortArray): ShortArray
//}
