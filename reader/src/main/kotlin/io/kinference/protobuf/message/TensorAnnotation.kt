package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TensorAnnotation(
    //ProtoTag = 1
    var tensor_name: String? = null,

    //ProtoTag = 2
    val quant_parameter_tensor_names: MutableList<StringStringEntryProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): TensorAnnotation {
            val proto = TensorAnnotation()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.tensor_name = reader.readString()
                    2 -> proto.quant_parameter_tensor_names.add(StringStringEntryProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
