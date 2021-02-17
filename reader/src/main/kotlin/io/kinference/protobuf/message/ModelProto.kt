package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class ModelProto(
    //ProtoTag = 1
    var ir_version: Long? = null,

    //ProtoTag = 8
    val opset_import: MutableList<OperatorSetIdProto> = ArrayList(),

    //ProtoTag = 2
    var producer_name: String? = null,

    //ProtoTag = 3
    var producer_version: String? = null,

    //ProtoTag = 4
    var domain: String? = null,

    //ProtoTag = 5
    var model_version: Long? = null,

    //ProtoTag = 6
    var doc_string: String? = null,

    //ProtoTag = 7
    var graph: GraphProto? = null,

    //ProtoTag = 14
    val metadata_props: MutableList<StringStringEntryProto> = ArrayList(),

    //ProtoTag = 20
    val training_info: MutableList<TrainingInfoProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): ModelProto {
            val proto = ModelProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.ir_version = reader.readLong()
                    8 -> proto.opset_import.add(OperatorSetIdProto.decode(reader))
                    2 -> proto.producer_name = reader.readString()
                    3 -> proto.producer_version = reader.readString()
                    4 -> proto.domain = reader.readString()
                    5 -> proto.model_version = reader.readLong()
                    6 -> proto.doc_string = reader.readString()
                    7 -> proto.graph = GraphProto.decode(reader)
                    14 -> proto.metadata_props.add(StringStringEntryProto.decode(reader))
                    20 -> proto.training_info.add(TrainingInfoProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
