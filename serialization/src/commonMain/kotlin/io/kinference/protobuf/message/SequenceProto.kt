package io.kinference.protobuf.message

import com.squareup.wire.*
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.readLongArray
import okio.Buffer

class SequenceProto(
    var name: String? = null,
    var elementType: DataType = DataType.UNDEFINED,
    var tensorValues: MutableList<TensorProto> = ArrayList(),
    var sequenceValues: MutableList<SequenceProto> = ArrayList(),
    var mapValues: MutableList<MapProto> = ArrayList()
) {
    private enum class ReaderTag(val tag: Int) {
        NAME(1),
        ELEM_TYPE(2),
        TENSOR_VALUES(3),
        SPARSE_TENSOR_VALUES(4),
        SEQUENCE_VALUES(5),
        MAP_VALUES(6),
        OPTIONAL_VALUES(7);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }

    companion object {
        fun decode(reader: ProtobufReader): SequenceProto {
            val proto = SequenceProto()
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.NAME -> proto.name = reader.readString()
                    ReaderTag.ELEM_TYPE -> proto.elementType = reader.readValue(DataType.ADAPTER)
                    ReaderTag.TENSOR_VALUES -> proto.tensorValues.add(TensorProto.decode(reader))
                    ReaderTag.SPARSE_TENSOR_VALUES -> error("Sparse tensor types are not supported")
                    ReaderTag.SEQUENCE_VALUES -> proto.sequenceValues.add(decode(reader))
                    ReaderTag.MAP_VALUES -> proto.mapValues.add(MapProto.decode(reader))
                    ReaderTag.OPTIONAL_VALUES -> error("Optional types are not supported")
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }

    enum class DataType(override val value: Int) : WireEnum {
        UNDEFINED(0),
        TENSOR(1),
        SPARSE_TENSOR(2),
        SEQUENCE(3),
        MAP(4),
        OPTIONAL(5);

        companion object {
            val ADAPTER: ProtoAdapter<DataType> = object : EnumAdapter<DataType>(DataType::class, Syntax.PROTO_2, UNDEFINED) {
                override fun fromValue(value: Int): DataType? = DataType.fromValue(value)
            }

            fun fromValue(value: Int): DataType? = when (value) {
                0 -> UNDEFINED
                1 -> TENSOR
                2 -> SPARSE_TENSOR
                3 -> SEQUENCE
                4 -> MAP
                5 -> OPTIONAL
                else -> null
            }
        }
    }
}
