package org.jetbrains.research.kotlin.mpp.inference.nn.parameters

import scientifik.kmath.linear.Point
import scientifik.kmath.linear.RealMatrix

sealed class Parameters

data class DenseParameters(val weights: RealMatrix?, val biases: RealMatrix?) : Parameters()

data class LSTMParameters(val Wf: RealMatrix?, val Wi: RealMatrix?, val Wo: RealMatrix?, val Wc: RealMatrix?,
                          val Uf: RealMatrix?, val Ui: RealMatrix?, val Uo: RealMatrix?, val Uc: RealMatrix?,
                          val bf: Point<Double>?, val bi: Point<Double>?, val bo: Point<Double>?, val bc: Point<Double>?) : Parameters()
