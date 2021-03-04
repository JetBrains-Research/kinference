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
        fun decode(reader: ProtobufReader): NodeProto {
            val proto = NodeProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.input.add(reader.readString())
                    2 -> proto.output.add(reader.readString())
                    3 -> proto.name = reader.readString()
                    4 -> proto.opType = reader.readString()
                    5 -> proto.attribute.add(AttributeProto.decode(reader))
                    6 -> reader.readString() // skip docstring
                    7 -> proto.domain = reader.readString()
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
