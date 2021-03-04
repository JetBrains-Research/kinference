package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class ValueInfoProto(
    val name: String? = null,
    val type: TypeProto? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): ValueInfoProto {
            var name: String? = null
            var type: TypeProto? = null
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> name = reader.readString()
                    2 -> type = TypeProto.decode(reader)
                    3 -> reader.readString() // skip docstring
                    else -> reader.readUnknownField(tag)
                }
            }
            return ValueInfoProto(name = name, type = type)
        }
    }
}
