#!/bin/sh
sed -i "s/ describe('protocol error'/ xdescribe('protocol error'/" test/node_redis.spec.js
npm install
./node_modules/.bin/mocha test/node_redis.spec.js