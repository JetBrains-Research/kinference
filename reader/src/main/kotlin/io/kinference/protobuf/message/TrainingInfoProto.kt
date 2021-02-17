package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TrainingInfoProto(
    //ProtoTag = 1
    var initialization: GraphProto? = null,

    //ProtoTag = 2
    var algorithm: GraphProto? = null,

    //ProtoTag = 3
    val initialization_binding: MutableList<StringStringEntryProto> = ArrayList(),

    //ProtoTag = 4
    val update_binding: MutableList<StringStringEntryProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): TrainingInfoProto {
            val proto = TrainingInfoProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.initialization = GraphProto.decode(reader)
                    2 -> proto.algorithm = GraphProto.decode(reader)
                    3 -> proto.initialization_binding.add(StringStringEntryProto.decode(reader))
                    4 -> proto.update_binding.add(StringStringEntryProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
