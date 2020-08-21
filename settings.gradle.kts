rootProject.name = "kotlin-inference"

include(":annotations", ":primitives-generator")

pluginManagement {
    repositories {
        gradlePluginPortal()
        jcenter()

        mavenCentral()

        maven("https://plugins.gradle.org/m2/")

        mavenLocal()
    }

    resolutionStrategy {
        eachPlugin {
            if (requested.id.id == "com.squareup.wire") {
                useModule("com.squareup.wire:wire-gradle-plugin:${requested.version}")
            }
        }
    }
}
