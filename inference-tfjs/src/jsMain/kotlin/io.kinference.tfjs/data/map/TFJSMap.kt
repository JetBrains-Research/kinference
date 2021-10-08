package io.kinference.tfjs.data.map

import io.kinference.data.ONNXDataType
import io.kinference.protobuf.message.MapProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.TFJSData
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.seq.TFJSSequence.Companion.extractTypeInfo
import io.kinference.tfjs.types.ValueInfo
import io.kinference.tfjs.types.ValueTypeInfo

class TFJSMap(data: Map<Any, TFJSData<*>>, info: ValueInfo) : TFJSData<Map<Any, TFJSData<*>>>(data, info) {
    override val type: ONNXDataType = ONNXDataType.ONNX_MAP

    val keyType: TensorProto.DataType
        get() = (info.typeInfo as ValueTypeInfo.MapTypeInfo).keyType

    val valueType: ValueTypeInfo
        get() = (info.typeInfo as ValueTypeInfo.MapTypeInfo).valueType

    override fun rename(name: String) = TFJSMap(data, ValueInfo(info.typeInfo, name))

    companion object {
        fun create(proto: MapProto): TFJSMap {
            val elementType = ValueTypeInfo.MapTypeInfo(proto.keyType, proto.values!!.extractTypeInfo())
            val info = ValueInfo(elementType, proto.name!!)
            val map = HashMap<Any, TFJSData<*>>().apply {
                val keys = if (proto.keyType == TensorProto.DataType.STRING) proto.stringKeys else castKeys(proto.keys!!, proto.keyType)
                keys.zip(TFJSSequence.create(proto.values!!).data) { key, value -> put(key, value) }
            }
            return TFJSMap(map, info)
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
