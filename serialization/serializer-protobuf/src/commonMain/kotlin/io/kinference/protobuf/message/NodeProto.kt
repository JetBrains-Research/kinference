package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class NodeProto(
    val input: MutableList<String> = ArrayList(),
    val output: MutableList<String> = ArrayList(),
    var name: String? = null,
    var opType: String? = null,
    var domain: String? = null,
    val attribute: MutableList<AttributeProto> = ArrayList()
) {
    companion object {
        suspend fun decode(reader: ProtobufReader): NodeProto {
            val proto = NodeProto()
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.INPUT -> proto.input.add(reader.readString())
                    ReaderTag.OUTPUT -> proto.output.add(reader.readString())
                    ReaderTag.NAME -> proto.name = reader.readString()
                    ReaderTag.OP_TYPE -> proto.opType = reader.readString()
                    ReaderTag.ATTRIBUTE -> proto.attribute.add(AttributeProto.decode(reader))
                    ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                    ReaderTag.DOMAIN -> proto.domain = reader.readString()
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        INPUT(1),
        OUTPUT(2),
        NAME(3),
        OP_TYPE(4),
        ATTRIBUTE(5),
        DOC_STRING(6),
        DOMAIN(7);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }
}
