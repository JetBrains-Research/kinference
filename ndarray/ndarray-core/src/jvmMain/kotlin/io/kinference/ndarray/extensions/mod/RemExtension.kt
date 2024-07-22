package io.kinference.ndarray.extensions.mod

import io.kinference.ndarray.arrays.NumberNDArrayCore

suspend operator fun NumberNDArrayCore.rem(other: NumberNDArrayCore) = fmod(other)
