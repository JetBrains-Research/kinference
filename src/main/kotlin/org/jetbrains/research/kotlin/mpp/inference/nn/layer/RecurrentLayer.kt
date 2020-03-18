package org.jetbrains.research.kotlin.mpp.inference.nn.layer

import org.jetbrains.research.kotlin.mpp.inference.nn.parameters.Parameters
import scientifik.kmath.structures.NDStructure

abstract class  RecurrentLayer<T : NDStructure<*>>(name: String, params: Parameters) : Layer<T>(name, params){

    abstract fun activate()
}
