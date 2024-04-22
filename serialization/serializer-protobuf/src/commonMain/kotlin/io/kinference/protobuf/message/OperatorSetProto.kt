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
        suspend fun decode(reader: ProtobufReader): OperatorSetProto {
            val proto = OperatorSetProto()
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.MAGIC -> proto.magic = reader.readString()
                    ReaderTag.IR_VERSION -> proto.irVersion = reader.readLong()
                    ReaderTag.IR_VERSION_PRE_RELEASE -> proto.irVersionPrerelease = reader.readString()
                    ReaderTag.DOMAIN -> proto.domain = reader.readString()
                    ReaderTag.OP_SET_VERSION -> proto.opSetVersion = reader.readLong()
                    ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                    ReaderTag.IR_BUILD_METADATA -> proto.irBuildMetadata = reader.readString()
                    ReaderTag.OPERATOR -> proto.operator.add(OperatorProto.decode(reader))
                    ReaderTag.FUNCTIONS -> proto.functions.add(FunctionProto.decode(reader))
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        MAGIC(1),
        IR_VERSION(2),
        IR_VERSION_PRE_RELEASE(3),
        DOMAIN(4),
        OP_SET_VERSION(5),
        DOC_STRING(6),
        IR_BUILD_METADATA(7),
        OPERATOR(8),
        FUNCTIONS(9);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }
}
