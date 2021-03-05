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
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.NAME -> name = reader.readString()
                    ReaderTag.TYPE -> type = TypeProto.decode(reader)
                    ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                }
            }
            return ValueInfoProto(name = name, type = type)
        }
    }

    private enum class ReaderTag(val tag: Int) {
        NAME(1),
        TYPE(2),
        DOC_STRING(3);

        companion object {
            fun fromInt(tag: Int) = values().first { it.tag == tag }
        }
    }
}
