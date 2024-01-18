package io.kinference.kmath

import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.*
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer

sealed class KIKMathData<T>(override val name: String?) : BaseONNXData<T> {
    abstract override fun rename(name: String): KIKMathData<T>
    abstract override fun clone(newName: String?): KIKMathData<T>

    class KMathTensor(name: String?, override val data: StructureND<*>) : KIKMathData<StructureND<*>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
        override fun rename(name: String): KMathTensor = KMathTensor(name, data)
        override fun clone(newName: String?): KMathTensor {
            return KMathTensor(newName, data.mapToBuffer { it!! })
        }
    }

    class KMathMap(name: String?, override val data: Map<Any, KIKMathData<*>>) : KIKMathData<Map<Any, KIKMathData<*>>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_MAP
        override fun rename(name: String): KMathMap = KMathMap(name, data)
        override fun clone(newName: String?): KMathMap {
            val newMap = HashMap<Any, KIKMathData<*>>(data.size)
            for ((key, value) in data.entries) {
                newMap[key] = value.clone()
            }
            return KMathMap(newName, newMap)
        }
    }

    class KMathSequence(name: String?, override val data: List<KIKMathData<*>>) : KIKMathData<List<KIKMathData<*>>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
        override fun rename(name: String): KMathSequence = KMathSequence(name, data)
        override fun clone(newName: String?): KMathSequence {
            return KMathSequence(newName, data.map { it.clone() })
        }
    }
}

object KIKMathTensorAdapter : ONNXDataAdapter<KIKMathData.KMathTensor, KITensor> {
    override fun fromONNXData(data: KITensor): KIKMathData.KMathTensor {
        val array = data.data
        val buffer = when (val type = data.data.type) {
            DataType.BYTE -> {
                val pointer = (array as ByteNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.SHORT -> {
                val pointer = (array as ShortNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.INT -> {
                val pointer = (array as IntNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.LONG -> {
                val pointer = (array as LongNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.UBYTE -> {
                val pointer = (array as UByteNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.USHORT -> {
                val pointer = (array as UShortNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.UINT -> {
                val pointer = (array as UIntNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.ULONG -> {
                val pointer = (array as ULongNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.FLOAT -> {
                val pointer = (array as FloatNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.DOUBLE -> {
                val pointer = (array as DoubleNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            DataType.BOOLEAN -> {
                val pointer = (array as BooleanNDArray).array.pointer()
                Buffer.auto(array.linearSize) { pointer.getAndIncrement() }
            }
            else -> error("Unsupported data type $type")
        }
        return KIKMathData.KMathTensor(data.name, BufferND(DefaultStrides(array.shape), buffer))
    }

    override fun toONNXData(data: KIKMathData.KMathTensor): KITensor {
        val elements = data.data.elements().map { it.second!! }.iterator()
        val shape = data.data.shape
        return when (val element = data.data.elements().first().second!!) {
            is Byte -> ByteNDArray(shape) { elements.next() as Byte }
            is Short -> ShortNDArray(shape) { elements.next() as Short }
            is Int -> IntNDArray(shape) { elements.next() as Int }
            is Long -> LongNDArray(shape) { elements.next() as Long }
            is UByte -> UByteNDArray(shape) { elements.next() as UByte }
            is UShort -> UShortNDArray(shape) { elements.next() as UShort }
            is UInt -> UIntNDArray(shape) { elements.next() as UInt }
            is ULong -> ULongNDArray(shape) { elements.next() as ULong }
            is Float -> FloatNDArray(shape) { elements.next() as Float }
            is Double -> DoubleNDArray(shape) { elements.next() as Double }
            is Boolean -> BooleanNDArray(shape) { elements.next() as Boolean }
            else -> error("Cannot convert from StructureND of ${element::class} to ONNXTensor")
        }.asTensor(data.name)
    }
}

object KIKMathMapAdapter : ONNXDataAdapter<KIKMathData.KMathMap, KIONNXMap> {
    override fun fromONNXData(data: KIONNXMap): KIKMathData.KMathMap {
        return KIKMathData.KMathMap(data.name, data.data.mapValues { it.value.toKIKMathData() })
    }

    override fun toONNXData(data: KIKMathData.KMathMap): KIONNXMap {
        val typeInfo = data.extractTypeInfo()
        val mapData = data.data.mapValues { it.value.toKIONNXData() } as Map<Any, KIONNXData<*>>
        return KIONNXMap(data.name, mapData, typeInfo as ValueTypeInfo.MapTypeInfo)
    }
}

object KIKMathSequenceAdapter : ONNXDataAdapter<KIKMathData.KMathSequence, KIONNXSequence> {
    override fun fromONNXData(data: KIONNXSequence): KIKMathData.KMathSequence {
        return KIKMathData.KMathSequence(data.name, data.data.map { it.toKIKMathData() })
    }

    override fun toONNXData(data: KIKMathData.KMathSequence): KIONNXSequence {
        val typeInfo = data.extractTypeInfo()
        val mapData = data.data.map { it.toKIONNXData() } as List<KIONNXData<*>>
        return KIONNXSequence(data.name, mapData, typeInfo as ValueTypeInfo.SequenceTypeInfo)
    }
}

fun KIKMathData<*>.toKIONNXData() = when (this.type) {
    ONNXDataType.ONNX_TENSOR -> KIKMathTensorAdapter.toONNXData(this as KIKMathData.KMathTensor)
    ONNXDataType.ONNX_SEQUENCE -> KIKMathSequenceAdapter.toONNXData(this as KIKMathData.KMathSequence)
    ONNXDataType.ONNX_MAP -> KIKMathMapAdapter.toONNXData(this as KIKMathData.KMathMap)
}

fun KIONNXData<*>.toKIKMathData() = when (this.type) {
    ONNXDataType.ONNX_TENSOR -> KIKMathTensorAdapter.fromONNXData(this as KITensor)
    ONNXDataType.ONNX_SEQUENCE -> KIKMathSequenceAdapter.fromONNXData(this as KIONNXSequence)
    ONNXDataType.ONNX_MAP -> KIKMathMapAdapter.fromONNXData(this as KIONNXMap)
}

fun KIKMathData<*>.extractTypeInfo(): ValueTypeInfo = when (this) {
    is KIKMathData.KMathTensor -> ValueTypeInfo.TensorTypeInfo(TensorShape(data.shape), data.elements().first().second.resolveProtoType())
    is KIKMathData.KMathSequence -> ValueTypeInfo.SequenceTypeInfo(data[0].extractTypeInfo())
    is KIKMathData.KMathMap -> {
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
    else -> error("Unsupported data type ${this!!::class}")
}
