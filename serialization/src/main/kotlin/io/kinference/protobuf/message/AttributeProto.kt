package io.kinference.protobuf.message

import com.squareup.wire.*
import io.kinference.protobuf.*
import okio.Buffer

class AttributeProto(
    var name: String? = null,
    var refAttrName: String? = null,
    var type: AttributeType = AttributeType.UNDEFINED,
    var f: Float? = null,
    var i: Long? = null,
    var s: String? = null,
    var t: TensorProto? = null,
    var g: GraphProto? = null,
    var sparseTensor: SparseTensorProto? = null,
    var floats: FloatArray? = null,
    var ints: LongArray? = null,
    val strings: MutableList<String> = ArrayList(),
    val tensors: MutableList<TensorProto> = ArrayList(),
    val graphs: MutableList<GraphProto> = ArrayList(),
    val sparseTensors: MutableList<SparseTensorProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): AttributeProto {
            val proto = AttributeProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.name = reader.readString()
                    21 -> proto.refAttrName = reader.readString()
                    13 -> reader.readString() // skip docstring
                    20 -> try {
                        proto.type = reader.readValue(AttributeType.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    2 -> proto.f = reader.readFloat()
                    3 -> proto.i = reader.readLong()
                    4 -> proto.s = reader.readBytes().utf8()
                    5 -> proto.t = TensorProto.decode(reader)
                    6 -> proto.g = GraphProto.decode(reader)
                    22 -> proto.sparseTensor = SparseTensorProto.decode(reader)
                    7 -> proto.floats = reader.readFloatArray(tag)
                    8 -> proto.ints = reader.readLongArray(tag)
                    9 -> proto.strings.add(reader.readBytes().utf8())
                    10 -> proto.tensors.add(TensorProto.decode(reader))
                    11 -> proto.graphs.add(GraphProto.decode(reader))
                    23 -> proto.sparseTensors.add(SparseTensorProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }

        fun decode(bytes: ByteArray) = decode(ProtobufReader(Buffer().write(bytes)))
    }

    enum class AttributeType(override val value: Int) : WireEnum {
        UNDEFINED(0),
        FLOAT(1),
        INT(2),
        STRING(3),
        TENSOR(4),
        GRAPH(5),
        SPARSE_TENSOR(11),
        FLOATS(6),
        INTS(7),
        STRINGS(8),
        TENSORS(9),
        GRAPHS(10),
        SPARSE_TENSORS(12);

        companion object {
            val ADAPTER: ProtoAdapter<AttributeType> = object : EnumAdapter<AttributeType>(AttributeType::class) {
                override fun fromValue(value: Int): AttributeType? = AttributeType.fromValue(value)
            }

            fun fromValue(value: Int): AttributeType? = when (value) {
                0 -> UNDEFINED
                1 -> FLOAT
                2 -> INT
                3 -> STRING
                4 -> TENSOR
                5 -> GRAPH
                6 -> FLOATS
                7 -> INTS
                8 -> STRINGS
                9 -> TENSORS
                10 -> GRAPHS
                11 -> SPARSE_TENSOR
                12 -> SPARSE_TENSORS
                else -> null
            }
        }
    }
}
