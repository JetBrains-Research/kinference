rootProject.name = "kotlin-inference"

include(":ndarray")

pluginManagement {
    repositories {
        gradlePluginPortal()
        jcenter()

        mavenCentral()

        maven("https://plugins.gradle.org/m2/")
    }

    resolutionStrategy {
        eachPlugin {
            if (requested.id.id == "com.squareup.wire") {
                useModule("com.squareup.wire:wire-gradle-plugin:${requested.version}")
            }
        }
    }
}
