@file:GeneratePrimitives(DataType.NUMBER)
package io.kinference.ndarray.extensions.constants

import io.kinference.primitives.annotations.*
import io.kinference.ndarray.toUShort
import io.kinference.ndarray.toUByte
import io.kinference.primitives.types.*


@GenerateNameFromPrimitives
@MakePublic
internal object PrimitiveConstants {
    val ONE = (1).toPrimitive()
    val ZERO = (0).toPrimitive()
    val MINUS_ONE = (-1).toPrimitive()
    val SQRT_1_2 = (0.7071067811865475).toPrimitive()
    val SQRT_2 = (1.41421356237309504880168872420969808).toPrimitive()
    val HALF = (0.5).toPrimitive()
    val ERF_P_VALUE = (0.3275911).toPrimitive()
    val ERF_COEF_1 = (0.254829592).toPrimitive()
    val ERF_COEF_2 = (-0.284496736).toPrimitive()
    val ERF_COEF_3 = (1.421413741).toPrimitive()
    val ERF_COEF_4 = (-1.453152027).toPrimitive()
    val ERF_COEF_5 = (1.061405429).toPrimitive()
    val TWO = (2.0).toPrimitive()
    val FGELU_COEF_1 = (0.035677408136300125).toPrimitive()
    val FGELU_COEF_2 = (0.7978845608028654).toPrimitive()

    val INV_ERF_COEF_1 = (4.330746750799873).toPrimitive()
    val INV_ERF_COEF_2 = (6.802721088435375).toPrimitive()

    val SIZE_BYTES = PrimitiveType.SIZE_BYTES.toLong()
}

