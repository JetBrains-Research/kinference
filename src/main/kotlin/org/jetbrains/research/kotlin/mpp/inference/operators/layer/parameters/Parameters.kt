package org.jetbrains.research.kotlin.mpp.inference.operators.layer.parameters

import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

sealed class Parameters<T : Number>

data class DataParameters<T : Number>(val weights: Tensor<T>, val biases: Tensor<T>) : Parameters<T>()

data class LSTMParameters<T : Number>(val inputGateParameters: GateParameters<T>,
                                      val forgetGateParameters: GateParameters<T>,
                                      val outputGateParameters: GateParameters<T>,
                                      val cellStateParameters: GateParameters<T>) : Parameters<T>()

typealias GateParameters<T> = DataParameters<T>
