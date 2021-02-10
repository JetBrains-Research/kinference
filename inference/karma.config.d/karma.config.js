config.middleware = config.middleware || [];
config.middleware.push('resource-loader');

function ResourceLoaderMiddleware() {
    const fs = require('fs');

    return function (request, response, next) {
        const uri = decodeURI(request.originalUrl)
        const content = fs.readFileSync(uri.startsWith('/absolute') ? uri.slice(9) : '../../../../inference' + uri);
        response.writeHead(200);
        response.end(content);
    }
}

config.plugins.push({
    'middleware:resource-loader': ['factory', ResourceLoaderMiddleware]
});

config.set({
  logLevel: config.LOG_DEBUG,
  client: {
    mocha: {
      timeout: 10000
    }
  }
})
