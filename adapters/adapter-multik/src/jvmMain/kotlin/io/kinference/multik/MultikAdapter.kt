package io.kinference.multik

import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

internal typealias MultikDataType = org.jetbrains.kotlinx.multik.ndarray.data.DataType

//TODO: support maps and sequences
object MultikAdapter : ONNXDataAdapter<MultiArray<Number, Dimension>, KITensor> {
    override fun toONNXData(name: String, data: MultiArray<Number, Dimension>): KITensor {
        val tiledArray = createArray(data.shape, data.data.data)
        return createNDArray(data.dtype.resolveKIDataType(), tiledArray, data.shape).asTensor(name)
    }

    override fun fromONNXData(data: KITensor): MultiArray<Number, Dimension> {
        val tensor = data.data as NumberNDArray
        return tensor.asMultiArray()
    }
}

fun MultikDataType.resolveKIDataType() = when (this) {
    MultikDataType.ByteDataType -> DataType.BYTE
    MultikDataType.ShortDataType -> DataType.SHORT
    MultikDataType.IntDataType -> DataType.INT
    MultikDataType.LongDataType -> DataType.LONG
    MultikDataType.FloatDataType -> DataType.FLOAT
    MultikDataType.DoubleDataType -> DataType.DOUBLE
}

fun DataType.resolveMultikDataType() = when (this) {
    DataType.BYTE -> MultikDataType.ByteDataType
    DataType.SHORT -> MultikDataType.ShortDataType
    DataType.INT -> MultikDataType.IntDataType
    DataType.LONG -> MultikDataType.LongDataType
    DataType.FLOAT -> MultikDataType.FloatDataType
    DataType.DOUBLE -> MultikDataType.DoubleDataType
    else -> error("$this type is not supported by Multik")
}


fun NumberNDArray.asMultiArray(): NDArray<Number, Dimension> {
    val dtype = type.resolveMultikDataType()
    val view = when (this) {
        is ByteNDArray -> MemoryViewByteArray(this.array.toArray())
        is ShortNDArray -> MemoryViewShortArray(this.array.toArray())
        is IntNDArray -> MemoryViewIntArray(this.array.toArray())
        is LongNDArray -> MemoryViewLongArray(this.array.toArray())
        is FloatNDArray -> MemoryViewFloatArray(this.array.toArray())
        is DoubleNDArray -> MemoryViewDoubleArray(this.array.toArray())
        else -> error("")
    } as MemoryView<Number>
    return NDArray(view, shape = shape, dtype = dtype, dim = dimensionOf(rank))
}
