package io.kinference.tfjs.types

import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.protobuf.message.TensorShapeProto
import io.kinference.protobuf.message.TypeProto
import io.kinference.tfjs.graph.Context

class TensorShape(private val dims: List<Dimension>) {
    constructor(shape: IntArray) : this(shape.map { StaticDimension(it) })

    val size: Int
        get() = dims.size

    open class Dimension
    class StaticDimension(val value: Int) : Dimension()
    class DynamicDimension(val value: String) : Dimension()
    object UnknownDimension : Dimension()

    fun getDimensions(context: Context? = null): IntArray {
        if (context == null) require(dims.all { it is StaticDimension })
        return dims.map {
            when (it) {
                is StaticDimension -> it.value
                is DynamicDimension -> context!!.getShape(it.value)
                is UnknownDimension -> -1
                else -> error("Unsupported dimension type")
            }
        }.toIntArray()
    }

    companion object {
        fun empty() = TensorShape(emptyList())

        operator fun invoke(proto: TensorShapeProto): TensorShape {
            return TensorShape(proto.dim.map {
                when {
                    it.dimValue != null -> StaticDimension(it.dimValue!!.toInt())
                    it.dimParam != null -> DynamicDimension(it.dimParam!!)
                    else -> UnknownDimension
                }
            })
        }
    }
}

sealed class ValueTypeInfo {
    companion object {
        fun create(proto: TypeProto) = when {
            proto.tensorType != null -> TensorTypeInfo(proto.tensorType!!)
            proto.sequenceType != null -> SequenceTypeInfo(proto.sequenceType!!)
            proto.mapType != null -> MapTypeInfo(proto.mapType!!)
            else -> error("One should be present")
        }
    }

    class TensorTypeInfo(val shape: TensorShape, val type: DataType) : ValueTypeInfo() {
        constructor(proto: TypeProto.Tensor) : this(proto.shape?.let { TensorShape(it) } ?: TensorShape.empty(), DataType.fromValue(proto.elem_type!!)!!)
    }

    class SequenceTypeInfo(val elementType: ValueTypeInfo) : ValueTypeInfo() {
        constructor(proto: TypeProto.Sequence) : this(create(proto.elem_type!!))
    }

    class MapTypeInfo(val keyType: DataType, val valueType: ValueTypeInfo) : ValueTypeInfo() {
        constructor(proto: TypeProto.Map) : this(DataType.fromValue(proto.key_type!!)!!, create(proto.value_type!!))
    }
}
