package io.kinference.algorithms.gec.tokenizer

import io.kinference.ndarray.arrays.IntMap
import io.kinference.ndarray.arrays.IntNDArray

data class TokenizedInputs(val inputsIds: IntNDArray, val padTokenId: Int){
    val attentionMask = inputsIds.map(object : IntMap {
        override fun apply(value: Int): Int {
            return if (value != padTokenId)
                1
            else
                0
        }
    })
}
