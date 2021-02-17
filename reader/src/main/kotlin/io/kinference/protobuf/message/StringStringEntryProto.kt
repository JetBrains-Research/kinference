package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class StringStringEntryProto(
    //ProtoTag = 1
    val key: String? = null,

    //ProtoTag = 2
    val value: String? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): StringStringEntryProto {
            var key: String? = null
            var value: String? = null
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> key = reader.readString()
                    2 -> value = reader.readString()
                    else -> reader.readUnknownField(tag)
                }
            }
            return StringStringEntryProto(key = key, value = value)
        }
    }
}
