config.middleware = config.middleware || [];
config.middleware.push('resource-loader');

function ResourceLoaderMiddleware() {
    const fs = require('fs');

    return function (request, response, next) {
        const content = fs.readFileSync('build/language-detector/processedResources/js' + decodeURI(request.originalUrl));
        response.writeHead(200);
        response.end(content);
    }
}

config.plugins.push({
    'middleware:resource-loader': ['factory', ResourceLoaderMiddleware]
});

config.set({
  client: {
    mocha: {
      timeout: 10000
    }
  }
})
