package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class NodeProto(
    //ProtoTag = 1
    val input: MutableList<String> = ArrayList(),

    //ProtoTag = 2
    val output: MutableList<String> = ArrayList(),

    //ProtoTag = 3
    var name: String? = null,

    //ProtoTag = 4
    var op_type: String? = null,

    //ProtoTag = 7
    var domain: String? = null,

    //ProtoTag = 5
    val attribute: MutableList<AttributeProto> = ArrayList(),

    //ProtoTag = 6
    var doc_string: String? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): NodeProto {
            val proto = NodeProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.input.add(reader.readString())
                    2 -> proto.output.add(reader.readString())
                    3 -> proto.name = reader.readString()
                    4 -> proto.op_type = reader.readString()
                    7 -> proto.domain = reader.readString()
                    5 -> proto.attribute.add(AttributeProto.decode(reader))
                    6 -> proto.doc_string = reader.readString()
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
