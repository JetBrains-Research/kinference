package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TrainingInfoProto(
    var initialization: GraphProto? = null,
    var algorithm: GraphProto? = null,
    val initializationBinding: MutableList<StringStringEntryProto> = ArrayList(),
    val updateBinding: MutableList<StringStringEntryProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): TrainingInfoProto {
            val proto = TrainingInfoProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.initialization = GraphProto.decode(reader)
                    2 -> proto.algorithm = GraphProto.decode(reader)
                    3 -> proto.initializationBinding.add(StringStringEntryProto.decode(reader))
                    4 -> proto.updateBinding.add(StringStringEntryProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
