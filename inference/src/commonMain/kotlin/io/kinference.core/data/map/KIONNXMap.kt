package io.kinference.core.data.map

import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.seq.KIONNXSequence.Companion.extractTypeInfo
import io.kinference.core.types.ValueInfo
import io.kinference.core.types.ValueTypeInfo
import io.kinference.data.ONNXData
import io.kinference.data.ONNXMap
import io.kinference.protobuf.message.*

class KIONNXMap(name: String?, data: Map<Any, ONNXData<*>>, val info: ValueTypeInfo.MapTypeInfo) : ONNXMap<Map<Any, ONNXData<*>>>(name, data) {
    constructor(data: Map<Any, ONNXData<*>>, info: ValueInfo) : this(info.name, data, info.typeInfo as ValueTypeInfo.MapTypeInfo)

    val keyType: TensorProto.DataType
        get() = info.keyType

    val valueType: ValueTypeInfo
        get() = info.valueType

    override fun rename(name: String): KIONNXMap = KIONNXMap(name, data, info)

    companion object {
        fun create(proto: MapProto): KIONNXMap {
            val elementType = ValueTypeInfo.MapTypeInfo(proto.keyType, proto.values!!.extractTypeInfo())
            val name = proto.name!!
            val map = HashMap<Any, ONNXData<*>>().apply {
                val keys = if (proto.keyType == TensorProto.DataType.STRING) proto.stringKeys else castKeys(proto.keys!!, proto.keyType)
                keys.zip(KIONNXSequence.create(proto.values!!).data) { key, value -> put(key, value) }
            }
            return KIONNXMap(name, map, elementType)
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
