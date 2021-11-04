package io.kinference.multik

import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.types.TensorShape
import io.kinference.core.types.ValueTypeInfo
import io.kinference.data.ONNXDataAdapter
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.createArray
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

internal typealias MultikDataType = org.jetbrains.kotlinx.multik.ndarray.data.DataType

object KIMultikTensorAdapter : ONNXDataAdapter<MultiArray<Number, Dimension>, KITensor> {
    override fun fromONNXData(data: KITensor): MultiArray<Number, Dimension> {
        val ndArray = data.data
        val dtype = ndArray.type.resolveMultikDataType()
        val view = when (val ndArray = data.data) {
            is ByteNDArray -> MemoryViewByteArray(ndArray.array.toArray())
            is ShortNDArray -> MemoryViewShortArray(ndArray.array.toArray())
            is IntNDArray -> MemoryViewIntArray(ndArray.array.toArray())
            is LongNDArray -> MemoryViewLongArray(ndArray.array.toArray())
            is FloatNDArray -> MemoryViewFloatArray(ndArray.array.toArray())
            is DoubleNDArray -> MemoryViewDoubleArray(ndArray.array.toArray())
            else -> error("${ndArray.type} type is not supported by Multik")
        } as MemoryView<Number>
        return NDArray(view, shape = ndArray.shape, dtype = dtype, dim = dimensionOf(ndArray.rank))
    }

    override fun toONNXData(name: String, data: MultiArray<Number, Dimension>): KITensor {
        val tiledArray = createArray(data.shape, data.data.data)
        return createNDArray(data.dtype.resolveKIDataType(), tiledArray, data.shape).asTensor(name)
    }
}

object KIMultikMapAdapter : ONNXDataAdapter<Map<Any, *>, KIONNXMap> {
    override fun fromONNXData(data: KIONNXMap): Map<Any, *> {
        return data.data.mapValues { it.value.fromKIONNXData() }
    }

    override fun toONNXData(name: String, data: Map<Any, *>): KIONNXMap {
        val typeInfo = data.extractTypeInfo()
        val mapData = data.mapValues { it.value.toKIONNXData(null) }
        return KIONNXMap(name, mapData, typeInfo as ValueTypeInfo.MapTypeInfo)
    }
}

object KIMultikSequenceAdapter : ONNXDataAdapter<List<*>, KIONNXSequence> {
    override fun fromONNXData(data: KIONNXSequence): List<*> {
        return data.data.map { it.fromKIONNXData() }
    }

    override fun toONNXData(name: String, data: List<*>): KIONNXSequence {
        val typeInfo = data.extractTypeInfo()
        val mapData = data.map { it.toKIONNXData(null) }
        return KIONNXSequence(name, mapData, typeInfo as ValueTypeInfo.SequenceTypeInfo)
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

private fun KIONNXData<*>.fromKIONNXData(): Any = when (this.type) {
    ONNXDataType.ONNX_TENSOR -> KIMultikTensorAdapter.fromONNXData(this as KITensor)
    ONNXDataType.ONNX_MAP -> KIMultikMapAdapter.fromONNXData(this as KIONNXMap)
    ONNXDataType.ONNX_SEQUENCE -> KIMultikSequenceAdapter.fromONNXData(this as KIONNXSequence)
}

private fun <T> T.toKIONNXData(name: String? = null): KIONNXData<*> = when (this) {
    is MultiArray<*, *> -> KIMultikTensorAdapter.toONNXData(name ?: "", this as MultiArray<Number, Dimension>)
    is List<*> -> KIMultikSequenceAdapter.toONNXData(name ?: "", this)
    is Map<*, *> -> KIMultikMapAdapter.toONNXData(name ?: "", this as Map<Any, *>)
    else -> error("Type info extraction failed. Cannot extract info from ${this!!::class}")
}

private fun <T> T.extractTypeInfo(): ValueTypeInfo = when (this) {
    is MultiArray<*, *> -> ValueTypeInfo.TensorTypeInfo(TensorShape(this.shape), this.data[0].resolveProtoType())
    is List<*> -> ValueTypeInfo.SequenceTypeInfo(this[0].extractTypeInfo())
    is Map<*, *> -> {
        val first = this.entries.first()
        ValueTypeInfo.MapTypeInfo(keyType = first.key.resolveProtoType(), valueType = first.value.extractTypeInfo())
    }
    else -> error("Type info extraction failed. Cannot extract info from ${this!!::class}")
}

private fun <T> T.resolveProtoType() = when (this) {
    is Byte -> TensorProto.DataType.INT8
    is Short -> TensorProto.DataType.INT16
    is Int -> TensorProto.DataType.INT32
    is Long -> TensorProto.DataType.INT64
    is UByte -> TensorProto.DataType.UINT8
    is UShort -> TensorProto.DataType.UINT16
    is UInt -> TensorProto.DataType.UINT32
    is ULong -> TensorProto.DataType.UINT64
    is Float -> TensorProto.DataType.FLOAT
    is Double -> TensorProto.DataType.DOUBLE
    is Boolean -> TensorProto.DataType.BOOL
    is String -> TensorProto.DataType.STRING
    else -> error("Cannot convert from StructureND of to ONNXTensor")
}
