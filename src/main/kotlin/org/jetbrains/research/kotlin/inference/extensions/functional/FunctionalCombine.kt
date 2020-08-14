package org.jetbrains.research.kotlin.inference.extensions.functional

fun interface PrimitiveArrayCombineFunction<T> {
    fun apply(array: T, arrayOffset: Int, otherArray: T, otherArrayOffset: Int, destinationArray: T, destinationArrayOffset: Int, size: Int): T
}

fun interface FloatArrayWithFloatArray : PrimitiveArrayCombineFunction<FloatArray> {
    override fun apply(array: FloatArray, arrayOffset: Int, otherArray: FloatArray, otherArrayOffset: Int, destinationArray: FloatArray, destinationArrayOffset: Int, size: Int): FloatArray
}

fun interface DoubleArrayWithDoubleArray : PrimitiveArrayCombineFunction<DoubleArray> {
    override fun apply(array: DoubleArray, arrayOffset: Int, otherArray: DoubleArray, otherArrayOffset: Int, destinationArray: DoubleArray, destinationArrayOffset: Int, size: Int): DoubleArray
}

fun interface IntArrayWithIntArray : PrimitiveArrayCombineFunction<IntArray> {
    override fun apply(array: IntArray, arrayOffset: Int, otherArray: IntArray, otherArrayOffset: Int, destinationArray: IntArray, destinationArrayOffset: Int, size: Int): IntArray
}

fun interface LongArrayWithLongArray : PrimitiveArrayCombineFunction<LongArray> {
    override fun apply(array: LongArray, arrayOffset: Int, otherArray: LongArray, otherArrayOffset: Int, destinationArray: LongArray, destinationArrayOffset: Int, size: Int): LongArray
}

fun interface ShortArrayWithShortArray : PrimitiveArrayCombineFunction<ShortArray> {
    override fun apply(array: ShortArray, arrayOffset: Int, otherArray: ShortArray, otherArrayOffset: Int, destinationArray: ShortArray, destinationArrayOffset: Int, size: Int): ShortArray
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
