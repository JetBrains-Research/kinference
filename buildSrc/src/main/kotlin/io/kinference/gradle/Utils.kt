package io.kinference.gradle

import org.jetbrains.kotlin.gradle.plugin.LanguageSettingsBuilder

//FIXME: In root gradle build file optIn does not resolve
fun LanguageSettingsBuilder.optInFixed(annotationName: String) = optIn(annotationName)
