package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class StringStringEntryProto(
    val key: String? = null,
    val value: String? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): StringStringEntryProto {
            var key: String? = null
            var value: String? = null
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.KEY -> key = reader.readString()
                    ReaderTag.VALUE -> value = reader.readString()
                    null -> reader.readUnknownField(tag)
                }
            }
            return StringStringEntryProto(key, value)
        }
    }

    private enum class ReaderTag(val tag: Int) {
        KEY(1),
        VALUE(2);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }
}
