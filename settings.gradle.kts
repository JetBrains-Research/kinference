rootProject.name = "kinference"

include(":loaders")
include(":inference")
include(":ndarray")

pluginManagement {
    repositories {
        gradlePluginPortal()
        jcenter()
        maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
    }

    resolutionStrategy {
        eachPlugin {
            if (requested.id.id == "com.squareup.wire") {
                useModule("com.squareup.wire:wire-gradle-plugin:${requested.version}")
            }
        }
    }
}
