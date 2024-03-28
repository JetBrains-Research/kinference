package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.readLongArray

class MapProto(
    var name: String? = null,
    var keyType: TensorProto.DataType = TensorProto.DataType.UNDEFINED,
    var keys: LongArray? = null,
    var stringKeys: ArrayList<String> = ArrayList(),
    var values: SequenceProto? = null
) {
    private enum class ReaderTag(val tag: Int) {
        NAME(1),
        KEY_TYPE(2),
        KEYS(3),
        STRING_KEYS(4),
        VALUES(5);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }

    companion object {
        suspend fun decode(reader: ProtobufReader): MapProto {
            val proto = MapProto()
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.NAME -> proto.name = reader.readString()
                    ReaderTag.KEY_TYPE -> proto.keyType = reader.readValue(TensorProto.DataType.ADAPTER)
                    ReaderTag.KEYS -> proto.keys = reader.readLongArray(tag)
                    ReaderTag.STRING_KEYS -> proto.stringKeys.add(reader.readString())
                    ReaderTag.VALUES -> proto.values = SequenceProto.decode(reader)
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
