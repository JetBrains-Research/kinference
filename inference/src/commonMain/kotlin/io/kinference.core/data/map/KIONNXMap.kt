package io.kinference.core.data.map

import io.kinference.core.data.KIONNXData
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.seq.KIONNXSequence.Companion.extractTypeInfo
import io.kinference.core.types.ValueInfo
import io.kinference.core.types.ValueTypeInfo
import io.kinference.data.ONNXDataType
import io.kinference.protobuf.message.*

class KIONNXMap(data: Map<Any, KIONNXData<*>>, info: ValueInfo) : KIONNXData<Map<Any, KIONNXData<*>>>(ONNXDataType.ONNX_MAP, data, info) {
    val keyType: TensorProto.DataType
        get() = (info.typeInfo as ValueTypeInfo.MapTypeInfo).keyType

    val valueType: ValueTypeInfo
        get() = (info.typeInfo as ValueTypeInfo.MapTypeInfo).valueType

    override fun rename(name: String): KIONNXMap = KIONNXMap(data, ValueInfo(info.typeInfo, name))

    companion object {
        fun create(proto: MapProto): KIONNXMap {
            val elementType = ValueTypeInfo.MapTypeInfo(proto.keyType, proto.values!!.extractTypeInfo())
            val info = ValueInfo(elementType, proto.name!!)
            val map = HashMap<Any, KIONNXData<*>>().apply {
                val keys = if (proto.keyType == TensorProto.DataType.STRING) proto.stringKeys else castKeys(proto.keys!!, proto.keyType)
                keys.zip(KIONNXSequence.create(proto.values!!).data) { key, value -> put(key, value) }
            }
            return KIONNXMap(map, info)
        }

        private fun castKeys(keys: LongArray, type: TensorProto.DataType) = when (type) {
            TensorProto.DataType.UINT8 -> keys.map { it.toUByte() }
            TensorProto.DataType.INT8 -> keys.map { it.toByte() }
            TensorProto.DataType.UINT16 -> keys.map { it.toUShort() }
            TensorProto.DataType.INT16 -> keys.map { it.toShort() }
            TensorProto.DataType.INT32 -> keys.map { it.toInt() }
            TensorProto.DataType.INT64 -> keys.map { it.toLong() }
            TensorProto.DataType.UINT32 -> keys.map { it.toUInt() }
            TensorProto.DataType.UINT64 -> keys.map { it.toULong() }
            else -> error("")
        }
    }
}
