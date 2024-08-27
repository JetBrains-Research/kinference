package io.kinference.ndarray.arrays.memory.storage

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.*
import io.kinference.primitives.types.DataType

internal interface TypedManualHandlingStorage {
    fun getNDArray(strides: Strides, fillZeros: Boolean = false, limiter: MemoryManager): MutableNDArrayCore
    fun returnNDArray(ndarray: NDArrayCore)
    fun clear()
}

interface ManualStorage : ArrayStorage {
    fun getNDArray(dataType: DataType, strides: Strides, fillZeros: Boolean = false): MutableNDArrayCore
    fun returnNDArray(ndArray: NDArrayCore)
}

internal class ManualArrayHandlingStorage(private val memoryManager: MemoryManager) : ManualStorage {
    private val storage: List<TypedManualHandlingStorage> = listOf(
        ByteManualHandlingArrayStorage(),
        ShortManualHandlingArrayStorage(),
        IntManualHandlingArrayStorage(),
        LongManualHandlingArrayStorage(),
        UByteManualHandlingArrayStorage(),
        UShortManualHandlingArrayStorage(),
        UIntManualHandlingArrayStorage(),
        ULongManualHandlingArrayStorage(),
        FloatManualHandlingArrayStorage(),
        DoubleManualHandlingArrayStorage(),
        BooleanManualHandlingArrayStorage()
    )

    override fun getNDArray(dataType: DataType, strides: Strides, fillZeros: Boolean): MutableNDArrayCore {
        return storage[dataType.ordinal].getNDArray(strides, fillZeros, memoryManager)
    }

    override fun returnNDArray(ndArray: NDArrayCore) {
        storage[ndArray.type.ordinal].returnNDArray(ndArray)
    }

    override fun resetState() {
        storage.forEach { it.clear() }
        memoryManager.resetLimit()
    }
}
