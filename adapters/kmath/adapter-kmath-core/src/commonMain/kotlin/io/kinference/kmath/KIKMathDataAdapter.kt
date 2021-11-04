package io.kinference.kmath

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
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer

object KIKMathTensorAdapter : ONNXDataAdapter<NDStructure<*>, KITensor> {
    override fun fromONNXData(data: KITensor): NDStructure<*> {
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
        return NDBuffer(DefaultStrides(array.shape), buffer)
    }

    override fun toONNXData(name: String, data: NDStructure<*>): KITensor {
        val elements = data.elements().map { it.second!! }.iterator()
        val shape = data.shape
        return when (val element = data.elements().first().second!!) {
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
        }.asTensor(name)
    }
}

object KIKMathMapAdapter : ONNXDataAdapter<Map<Any, *>, KIONNXMap> {
    override fun fromONNXData(data: KIONNXMap): Map<Any, *> {
        return data.data.mapValues { it.value.fromKIONNXData() }
    }

    override fun toONNXData(name: String, data: Map<Any, *>): KIONNXMap {
        val typeInfo = data.extractTypeInfo()
        val mapData = data.mapValues { it.value.toKIONNXData(null) }
        return KIONNXMap(name, mapData, typeInfo as ValueTypeInfo.MapTypeInfo)
    }
}

object KIKMathSequenceAdapter : ONNXDataAdapter<List<*>, KIONNXSequence> {
    override fun fromONNXData(data: KIONNXSequence): List<*> {
        return data.data.map { it.fromKIONNXData() }
    }

    override fun toONNXData(name: String, data: List<*>): KIONNXSequence {
        val typeInfo = data.extractTypeInfo()
        val mapData = data.map { it.toKIONNXData(null) }
        return KIONNXSequence(name, mapData, typeInfo as ValueTypeInfo.SequenceTypeInfo)
    }
}

private fun <T> T.toKIONNXData(name: String? = null): KIONNXData<*> = when (this) {
    is NDStructure<*> -> KIKMathTensorAdapter.toONNXData(name ?: "", this)
    is List<*> -> KIKMathSequenceAdapter.toONNXData(name ?: "", this)
    is Map<*, *> -> KIKMathMapAdapter.toONNXData(name ?: "", this as Map<Any, *>)
    else -> error("Data conversion from ${this!!::class} to ONNX format failed")
}

private fun KIONNXData<*>.fromKIONNXData(): Any = when (this.type) {
    ONNXDataType.ONNX_TENSOR -> KIKMathTensorAdapter.fromONNXData(this as KITensor)
    ONNXDataType.ONNX_SEQUENCE -> KIKMathSequenceAdapter.fromONNXData(this as KIONNXSequence)
    ONNXDataType.ONNX_MAP -> KIKMathMapAdapter.fromONNXData(this as KIONNXMap)
}

private fun <T> T.extractTypeInfo(): ValueTypeInfo = when (this) {
    is NDStructure<*> -> ValueTypeInfo.TensorTypeInfo(TensorShape(this.shape), this.elements().first().second.resolveProtoType())
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
    else -> error("Unsupported data type ${this!!::class}")
}
