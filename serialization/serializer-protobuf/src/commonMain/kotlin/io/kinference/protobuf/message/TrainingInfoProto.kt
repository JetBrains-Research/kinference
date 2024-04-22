package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TrainingInfoProto(
    var initialization: GraphProto? = null,
    var algorithm: GraphProto? = null,
    val initializationBinding: MutableList<StringStringEntryProto> = ArrayList(),
    val updateBinding: MutableList<StringStringEntryProto> = ArrayList()
) {
    companion object {
        suspend fun decode(reader: ProtobufReader): TrainingInfoProto {
            val proto = TrainingInfoProto()
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.INITIALIZATION -> proto.initialization = GraphProto.decode(reader)
                    ReaderTag.ALGORITHM -> proto.algorithm = GraphProto.decode(reader)
                    ReaderTag.INIT_BINDING -> proto.initializationBinding.add(StringStringEntryProto.decode(reader))
                    ReaderTag.UPDATE_BINDING -> proto.updateBinding.add(StringStringEntryProto.decode(reader))
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        INITIALIZATION(1),
        ALGORITHM(2),
        INIT_BINDING(3),
        UPDATE_BINDING(4);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }
}
