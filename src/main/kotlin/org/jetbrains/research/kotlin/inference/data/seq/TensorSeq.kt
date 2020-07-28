package org.jetbrains.research.kotlin.inference.data.seq

import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.ONNXDataType
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.types.SequenceInfo

class TensorSeq(val data: List<Tensor>, info: SequenceInfo) : ONNXData(ONNXDataType.ONNX_SEQUENCE, info) {
    override fun clone(newName: String): ONNXData = TensorSeq(data, info as SequenceInfo)

    val length: Int
        get() = data.size
}
