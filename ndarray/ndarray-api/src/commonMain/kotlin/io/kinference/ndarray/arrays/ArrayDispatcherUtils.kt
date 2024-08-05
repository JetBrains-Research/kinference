package io.kinference.ndarray.arrays

typealias StateMarker = () -> Unit

enum class ArrayTypes(val index: Int, val size: Int) {
    ByteArray(0, Byte.SIZE_BYTES),
    UByteArray(1, UByte.SIZE_BYTES),
    ShortArray(2, Short.SIZE_BYTES),
    UShortArray(3, UShort.SIZE_BYTES),
    IntArray(4, Int.SIZE_BYTES),
    UIntArray(5, UInt.SIZE_BYTES),
    LongArray(6, Long.SIZE_BYTES),
    ULongArray(7, ULong.SIZE_BYTES),
    FloatArray(8, Float.SIZE_BYTES),
    DoubleArray(9, Double.SIZE_BYTES),
    BooleanArray(10, 1);
}
