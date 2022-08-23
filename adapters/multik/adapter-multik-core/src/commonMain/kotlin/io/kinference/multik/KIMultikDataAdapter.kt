package multik

import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo
import io.kinference.data.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.ndarray.extensions.tiledFromPrimitiveArray
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.first

internal typealias MultikDataType = org.jetbrains.kotlinx.multik.ndarray.data.DataType

sealed class KIMultikData<T>(override val name: String?) : BaseONNXData<T> {
    class MultikTensor(name: String?, override val data: MultiArray<Number, Dimension>) : KIMultikData<MultiArray<Number, Dimension>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
        override fun rename(name: String): KIMultikData<MultiArray<Number, Dimension>> = MultikTensor(name, data)
    }

    class MultikMap(name: String?, override val data: Map<Any, KIMultikData<*>>) : KIMultikData<Map<Any, KIMultikData<*>>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_MAP
        override fun rename(name: String): KIMultikData<Map<Any, KIMultikData<*>>> = MultikMap(name, data)
    }

    class MultikSequence(name: String?, override val data: List<KIMultikData<*>>) : KIMultikData<List<KIMultikData<*>>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
        override fun rename(name: String): KIMultikData<List<KIMultikData<*>>> = MultikSequence(name, data)
    }
}

object KIMultikTensorAdapter : ONNXDataAdapter<KIMultikData.MultikTensor, KITensor> {
    override fun fromONNXData(data: KITensor): KIMultikData.MultikTensor {
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
        val multikArray = NDArray(view, shape = ndArray.shape/*, dtype = dtype*/, dim = dimensionOf(ndArray.rank))
        return KIMultikData.MultikTensor(data.name, multikArray)
    }

    override fun toONNXData(data: KIMultikData.MultikTensor): KITensor {
        val tiledArray = tiledFromPrimitiveArray(data.data.shape, data.data.data.data)
        return createNDArray(data.data.dtype.resolveKIDataType(), tiledArray, data.data.shape).asTensor(data.name)
    }
}

object KIMultikMapAdapter : ONNXDataAdapter<KIMultikData.MultikMap, KIONNXMap> {
    override fun fromONNXData(data: KIONNXMap): KIMultikData.MultikMap {
        return KIMultikData.MultikMap(data.name, data.data.mapValues { it.value.fromKIONNXData() })
    }

    override fun toONNXData(data: KIMultikData.MultikMap): KIONNXMap {
        val typeInfo = data.extractTypeInfo()
        val mapData = data.data.mapValues { it.value.toKIONNXData() }
        return KIONNXMap(data.name, mapData, typeInfo as ValueTypeInfo.MapTypeInfo)
    }
}

object KIMultikSequenceAdapter : ONNXDataAdapter<KIMultikData.MultikSequence, KIONNXSequence> {
    override fun fromONNXData(data: KIONNXSequence): KIMultikData.MultikSequence {
        return KIMultikData.MultikSequence(data.name, data.data.map { it.fromKIONNXData() })
    }

    override fun toONNXData(data: KIMultikData.MultikSequence): KIONNXSequence {
        val typeInfo = data.extractTypeInfo()
        val mapData = data.data.map { it.toKIONNXData() }
        return KIONNXSequence(data.name, mapData, typeInfo as ValueTypeInfo.SequenceTypeInfo)
    }
}

fun MultikDataType.resolveKIDataType() = when (this) {
    MultikDataType.ByteDataType -> DataType.BYTE
    MultikDataType.ShortDataType -> DataType.SHORT
    MultikDataType.IntDataType -> DataType.INT
    MultikDataType.LongDataType -> DataType.LONG
    MultikDataType.FloatDataType -> DataType.FLOAT
    MultikDataType.DoubleDataType -> DataType.DOUBLE
    else -> error("Unknown multik data type")
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

fun KIONNXData<*>.fromKIONNXData() = when (this.type) {
    ONNXDataType.ONNX_TENSOR -> KIMultikTensorAdapter.fromONNXData(this as KITensor)
    ONNXDataType.ONNX_MAP -> KIMultikMapAdapter.fromONNXData(this as KIONNXMap)
    ONNXDataType.ONNX_SEQUENCE -> KIMultikSequenceAdapter.fromONNXData(this as KIONNXSequence)
}

fun KIMultikData<*>.toKIONNXData() = when (this.type) {
    ONNXDataType.ONNX_TENSOR -> KIMultikTensorAdapter.toONNXData(this as KIMultikData.MultikTensor)
    ONNXDataType.ONNX_MAP -> KIMultikMapAdapter.toONNXData(this as KIMultikData.MultikMap)
    ONNXDataType.ONNX_SEQUENCE -> KIMultikSequenceAdapter.toONNXData(this as KIMultikData.MultikSequence)
}

fun KIMultikData<*>.extractTypeInfo(): ValueTypeInfo = when (this) {
    is KIMultikData.MultikTensor -> ValueTypeInfo.TensorTypeInfo(TensorShape(data.shape), data.first().resolveProtoType())
    is KIMultikData.MultikSequence -> ValueTypeInfo.SequenceTypeInfo(data[0].extractTypeInfo())
    is KIMultikData.MultikMap -> {
        val first = data.entries.first()
        ValueTypeInfo.MapTypeInfo(keyType = first.key.resolveProtoType(), valueType = first.value.extractTypeInfo())
    }
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
