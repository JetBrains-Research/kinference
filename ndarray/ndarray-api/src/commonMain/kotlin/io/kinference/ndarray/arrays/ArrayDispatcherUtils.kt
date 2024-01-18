package io.kinference.ndarray.arrays

typealias StateMarker = (ArrayUsageMarker) -> Unit

enum class ArrayUsageMarker {
    Unused,
    Used,
    ContextOutput,
    GlobalOutput
}

class ArrayContainer<T>(val array: T, var marker: ArrayUsageMarker = ArrayUsageMarker.Used) {
    val markAsOutput: StateMarker = {
        marker = it
    }
}

enum class ArrayTypes(val index: Int, val initializer: (Int) -> ArrayContainer<*>) {
    ByteArray(0, { size -> ArrayContainer(ByteArray(size)) }),
    UByteArray(1, { size -> ArrayContainer(UByteArray(size)) }),
    ShortArray(2, { size -> ArrayContainer(ShortArray(size)) }),
    UShortArray(3, { size -> ArrayContainer(UShortArray(size)) }),
    IntArray(4, { size -> ArrayContainer(IntArray(size)) }),
    UIntArray(5, { size -> ArrayContainer(UIntArray(size)) }),
    LongArray(6, { size -> ArrayContainer(LongArray(size)) }),
    ULongArray(7, { size -> ArrayContainer(ULongArray(size)) }),
    FloatArray(8, { size -> ArrayContainer(FloatArray(size)) }),
    DoubleArray(9, { size -> ArrayContainer(DoubleArray(size)) }),
    CharArray(10, { size -> ArrayContainer(CharArray(size)) }),
    BooleanArray(11, { size -> ArrayContainer(BooleanArray(size)) });

    fun createArray(size: Int) : ArrayContainer<*> {
        return initializer(size)
    }
}

fun resetPrimitiveArray(array: ArrayContainer<*>) {
    when (val arr = array.array!!) {
        is ByteArray -> arr.fill(0)       // 8-bit signed
        is UByteArray -> arr.fill(0u)     // 8-bit unsigned
        is ShortArray -> arr.fill(0)      // 16-bit signed
        is UShortArray -> arr.fill(0u)    // 16-bit unsigned
        is IntArray -> arr.fill(0)        // 32-bit signed
        is UIntArray -> arr.fill(0u)      // 32-bit unsigned
        is LongArray -> arr.fill(0L)      // 64-bit signed
        is ULongArray -> arr.fill(0U)     // 64-bit unsigned
        is FloatArray -> arr.fill(0.0f)
        is DoubleArray -> arr.fill(0.0)
        is CharArray -> arr.fill('\u0000')
        is BooleanArray -> arr.fill(false)
        else -> throw IllegalArgumentException("Unsupported array type")
    }
}
