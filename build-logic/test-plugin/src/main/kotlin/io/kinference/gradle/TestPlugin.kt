package io.kinference.gradle

import org.gradle.api.Plugin
import org.gradle.api.Project

/**
 * Plugin that provides test configuration for KInference projects.
 * This plugin applies the test configuration functions to Kotlin Multiplatform projects.
 */
class TestPlugin : Plugin<Project> {
    override fun apply(project: Project) {
        // The plugin doesn't need to do anything on apply
        // The test configuration functions are extension functions that can be called directly
        // on KotlinJvmTarget and KotlinJsTargetDsl instances
    }
}
