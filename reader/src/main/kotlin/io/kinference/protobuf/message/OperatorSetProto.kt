package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class OperatorSetProto(
    //ProtoTag = 1
    var magic: String? = null,

    //ProtoTag = 2
    var ir_version: Long? = null,

    //ProtoTag = 3
    var ir_version_prerelease: String? = null,

    //ProtoTag = 7
    var ir_build_metadata: String? = null,

    //ProtoTag = 4
    var domain: String? = null,

    //ProtoTag = 5
    var opset_version: Long? = null,

    //ProtoTag = 6
    var doc_string: String? = null,

    //ProtoTag = 8
    val operator: MutableList<OperatorProto> = ArrayList(),

    //ProtoTag = 9
    val functions: MutableList<FunctionProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): OperatorSetProto {
            val proto = OperatorSetProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.magic = reader.readString()
                    2 -> proto.ir_version = reader.readLong()
                    3 -> proto.ir_version_prerelease = reader.readString()
                    7 -> proto.ir_build_metadata = reader.readString()
                    4 -> proto.domain = reader.readString()
                    5 -> proto.opset_version = reader.readLong()
                    6 -> proto.doc_string = reader.readString()
                    8 -> proto.operator.add(OperatorProto.decode(reader))
                    9 -> proto.functions.add(FunctionProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
