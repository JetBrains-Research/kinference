package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import kotlin.coroutines.CoroutineContext

data class ManualAllocatorContext internal constructor(private val storage: SingleArrayStorage) : CoroutineContext.Element {

    companion object Key : CoroutineContext.Key<ManualAllocatorContext>
    override val key: CoroutineContext.Key<*> get() = Key

    fun getNDArray(dataType: DataType, strides: Strides, fillZeros: Boolean = false): MutableNDArrayCore {
        return when(dataType) {
            DataType.BYTE -> ByteArrayStorageWrapper.getNDArray(strides, storage, fillZeros)
            DataType.SHORT -> ShortArrayStorageWrapper.getNDArray(strides, storage, fillZeros)
            DataType.INT -> IntArrayStorageWrapper.getNDArray(strides, storage, fillZeros)
            DataType.LONG -> LongArrayStorageWrapper.getNDArray(strides, storage, fillZeros)

            DataType.UBYTE -> UByteArrayStorageWrapper.getNDArray(strides, storage, fillZeros)
            DataType.USHORT -> UShortArrayStorageWrapper.getNDArray(strides, storage, fillZeros)
            DataType.UINT -> UIntArrayStorageWrapper.getNDArray(strides, storage, fillZeros)
            DataType.ULONG -> ULongArrayStorageWrapper.getNDArray(strides, storage, fillZeros)

            DataType.FLOAT -> FloatArrayStorageWrapper.getNDArray(strides, storage, fillZeros)
            DataType.DOUBLE -> DoubleArrayStorageWrapper.getNDArray(strides, storage, fillZeros)

            DataType.BOOLEAN -> BooleanArrayStorageWrapper.getNDArray(strides, storage, fillZeros)

            else -> error("Unsupported array type")
        }
    }

    fun returnNDArray(ndArray: NDArrayCore) {
        when(ndArray.type) {
            DataType.BYTE -> ByteArrayStorageWrapper.returnNDArray(storage, ndArray as ByteNDArray)
            DataType.SHORT -> ShortArrayStorageWrapper.returnNDArray(storage, ndArray as ShortNDArray)
            DataType.INT -> IntArrayStorageWrapper.returnNDArray(storage, ndArray as IntNDArray)
            DataType.LONG -> LongArrayStorageWrapper.returnNDArray(storage, ndArray as LongNDArray)

            DataType.UBYTE -> UByteArrayStorageWrapper.returnNDArray(storage, ndArray as UByteNDArray)
            DataType.USHORT -> UShortArrayStorageWrapper.returnNDArray(storage, ndArray as UShortNDArray)
            DataType.UINT -> UIntArrayStorageWrapper.returnNDArray(storage, ndArray as UIntNDArray)
            DataType.ULONG -> ULongArrayStorageWrapper.returnNDArray(storage, ndArray as ULongNDArray)

            DataType.FLOAT -> FloatArrayStorageWrapper.returnNDArray(storage, ndArray as FloatNDArray)
            DataType.DOUBLE -> DoubleArrayStorageWrapper.returnNDArray(storage, ndArray as DoubleNDArray)

            DataType.BOOLEAN -> BooleanArrayStorageWrapper.returnNDArray(storage, ndArray as BooleanNDArray)

            else -> error("Unsupported array type")
        }
    }
}
