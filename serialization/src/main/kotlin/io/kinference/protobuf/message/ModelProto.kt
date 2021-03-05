package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class ModelProto(
    var irVersion: Long? = null,
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
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.IR_VERSION -> proto.irVersion = reader.readLong()
                    ReaderTag.PRODUCER_NAME -> proto.producerName = reader.readString()
                    ReaderTag.PRODUCER_VERSION -> proto.producerVersion = reader.readString()
                    ReaderTag.DOMAIN -> proto.domain = reader.readString()
                    ReaderTag.MODEL_VERSION -> proto.modelVersion = reader.readLong()
                    ReaderTag.DOC_STRING -> reader.readString() //skip docstring
                    ReaderTag.GRAPH -> proto.graph = GraphProto.decode(reader)
                    ReaderTag.OP_SET_IMPORT -> proto.opSetImport.add(OperatorSetIdProto.decode(reader))
                    ReaderTag.METADATA_PROPS -> proto.metadataProps.add(StringStringEntryProto.decode(reader))
                    ReaderTag.TRAINING_INFO -> proto.trainingInfo.add(TrainingInfoProto.decode(reader))
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        IR_VERSION(1),
        PRODUCER_NAME(2),
        PRODUCER_VERSION(3),
        DOMAIN(4),
        MODEL_VERSION(5),
        DOC_STRING(6),
        GRAPH(7),
        OP_SET_IMPORT(8),
        METADATA_PROPS(14),
        TRAINING_INFO(20);

        companion object {
            fun fromInt(tag: Int) = values().first { it.tag == tag }
        }
    }
}
