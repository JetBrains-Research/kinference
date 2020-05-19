package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.Operator

object LSTMFactory {
    inline fun <reified T : Number> create(attributes: Map<String, Attribute<*>>) : Operator<T>{
        return when(attributes["direction"]?.value){
            "forward" -> LSTMLayer()
            "bidirectional" -> BiLSTMLayer()
            else -> LSTMLayer()
        }
    }
}

