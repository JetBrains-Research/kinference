package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class ModelProto(
    var ir_version: Long? = null,
    val opSetImport: MutableList<OperatorSetIdProto> = ArrayList(),
    var producerName: String? = null,
    var producerVersion: String? = null,
    var domain: String? = null,
    var modelVersion: Long? = null,
    var graph: GraphProto? = null,
    val metadataProps: MutableList<StringStringEntryProto> = ArrayList(),
    val trainingInfo: MutableList<TrainingInfoProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): ModelProto {
            val proto = ModelProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.ir_version = reader.readLong()
                    2 -> proto.producerName = reader.readString()
                    3 -> proto.producerVersion = reader.readString()
                    4 -> proto.domain = reader.readString()
                    5 -> proto.modelVersion = reader.readLong()
                    6 -> reader.readString() //skip docstring
                    7 -> proto.graph = GraphProto.decode(reader)
                    8 -> proto.opSetImport.add(OperatorSetIdProto.decode(reader))
                    14 -> proto.metadataProps.add(StringStringEntryProto.decode(reader))
                    20 -> proto.trainingInfo.add(TrainingInfoProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
