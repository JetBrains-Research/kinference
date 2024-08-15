package io.kinference.ndarray.arrays

enum class ArrayTypes(val index: Int, val size: Int) {
    ByteArrayType(0, Byte.SIZE_BYTES),
    UByteArrayType(1, UByte.SIZE_BYTES),
    ShortArrayType(2, Short.SIZE_BYTES),
    UShortArrayType(3, UShort.SIZE_BYTES),
    IntArrayType(4, Int.SIZE_BYTES),
    UIntArrayType(5, UInt.SIZE_BYTES),
    LongArrayType(6, Long.SIZE_BYTES),
    ULongArrayType(7, ULong.SIZE_BYTES),
    FloatArrayType(8, Float.SIZE_BYTES),
    DoubleArrayType(9, Double.SIZE_BYTES),
    BooleanArrayType(10, 1);

    companion object {
        fun sizeInBytes(index: Int, arraySize: Int): Long = entries[index].size * arraySize.toLong()
    }
}
