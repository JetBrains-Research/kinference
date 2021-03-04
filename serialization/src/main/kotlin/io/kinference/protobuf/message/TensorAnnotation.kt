package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TensorAnnotation(
    var tensorName: String? = null,
    val quantParameterTensorNames: MutableList<StringStringEntryProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): TensorAnnotation {
            val proto = TensorAnnotation()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.tensorName = reader.readString()
                    2 -> proto.quantParameterTensorNames.add(StringStringEntryProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
