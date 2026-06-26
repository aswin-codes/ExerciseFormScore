const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  resolver: {
    assetExts: [...defaultConfig.resolver.assetExts, 'tflite'],
    blockList: [
      /[\/\\]node_modules[\/\\]react-native-fast-tflite[\/\\]android[\/\\]/,
    ].concat(defaultConfig.resolver.blockList || []),
  },
};

module.exports = mergeConfig(defaultConfig, config);
