const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const exclusionList = require('metro-config/src/defaults/exclusionList');
/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  resolver: {
      assetExts: [...getDefaultConfig(__dirname).resolver.assetExts, 'tflite'],
      blockList: exclusionList([
        /.*\/node_modules\/react-native-fast-tflite\/android\/.*/,
      ]),
    },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
