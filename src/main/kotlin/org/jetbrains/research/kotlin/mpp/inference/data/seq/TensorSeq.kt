package org.jetbrains.research.kotlin.mpp.inference.data.seq

import org.jetbrains.research.kotlin.mpp.inference.data.ONNXData
import org.jetbrains.research.kotlin.mpp.inference.types.SequenceInfo
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXDataType
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor

class TensorSeq(val data: List<Tensor>, info: SequenceInfo) : ONNXData(ONNXDataType.ONNX_SEQUENCE, info) {
    override fun clone(newName: String): ONNXData = TensorSeq(data, info as SequenceInfo)

    val length: Int
        get() = data.size
}
