package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class OperatorSetProto(
    var magic: String? = null,
    var irVersion: Long? = null,
    var irVersionPrerelease: String? = null,
    var irBuildMetadata: String? = null,
    var domain: String? = null,
    var opSetVersion: Long? = null,
    val operator: MutableList<OperatorProto> = ArrayList(),
    val functions: MutableList<FunctionProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): OperatorSetProto {
            val proto = OperatorSetProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.magic = reader.readString()
                    2 -> proto.irVersion = reader.readLong()
                    3 -> proto.irVersionPrerelease = reader.readString()
                    4 -> proto.domain = reader.readString()
                    5 -> proto.opSetVersion = reader.readLong()
                    6 -> reader.readString() // skip docstring
                    7 -> proto.irBuildMetadata = reader.readString()
                    8 -> proto.operator.add(OperatorProto.decode(reader))
                    9 -> proto.functions.add(FunctionProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
