package io.kinference.ndarray.extensions.abs

import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.ndarray.arrays.*
import kotlin.jvm.JvmName

//@JvmName("absExtension")
//fun NumberNDArrayCore.abs() = abs(this)

fun abs(x: NumberNDArrayCore): NumberNDArrayCore {
    return when (x) {
        is UIntNDArray, is UShortNDArray, is UByteNDArray, is ULongNDArray -> x
        is FloatNDArray -> absFloat(x)
        is DoubleNDArray -> absDouble(x)
        is IntNDArray -> absInt(x)
        is LongNDArray -> absLong(x)
        is ShortNDArray -> absShort(x)
        is ByteNDArray -> absByte(x)
        else -> error("Abs isn't support NDArray with type ${x.type}")
    }
}
