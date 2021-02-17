package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class ValueInfoProto(
    //ProtoTag = 1
    val name: String? = null,

    //ProtoTag = 2
    val type: TypeProto? = null,

    //ProtoTag = 3
    val doc_string: String? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): ValueInfoProto {
            var name: String? = null
            var type: TypeProto? = null
            var doc_string: String? = null
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> name = reader.readString()
                    2 -> type = TypeProto.decode(reader)
                    3 -> doc_string = reader.readString()
                    else -> reader.readUnknownField(tag)
                }
            }
            return ValueInfoProto(name = name, type = type, doc_string = doc_string)
        }
    }
}
